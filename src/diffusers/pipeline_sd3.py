"""SD3.5 Medium counterpart to ``pipeline.py``'s ``ModifiedStableDiffusionImg2ImgPipeline``.

Same classifier-guided cone-projection counterfactual steering (Dhariwal &
Nichol-style classifier guidance, projected onto the CFG direction and
renormalized -- see ``cc_ddim.py`` / ``pipeline.py`` for the DDIM version),
ported from SD1.x's epsilon-prediction + DDIM math to SD3.5's rectified-flow
formulation. The two are NOT the same schedule, so this is a real port, not a
copy-paste:

* SD1.x: ``x_t = sqrt(a_t) x0 + sqrt(1-a_t) eps``, model predicts ``eps``,
  classifier-guidance scales by ``sqrt(1 - a_t)``.
* SD3.5 (rectified flow): ``x_t = (1 - sigma) x0 + sigma * noise``, the
  transformer predicts the velocity ``v = noise - x0`` directly, so
  ``x0 = x_t - sigma * v``. By the same "coefficient of the model's direct
  output in the x_t parameterization, holding x0 fixed" derivation that gives
  SD1.x its ``sqrt(1-a_t)`` factor, the flow-matching analogue is simply
  ``sigma`` -- ``d(x_t)/d(v) = sigma`` where SD1.x has
  ``d(x_t)/d(eps) = sqrt(1-a_t)``. That substitution (``sigma`` in place of
  ``(1 - a_t).sqrt()``) is the one piece of this port that's a principled
  derivation rather than a verified-against-ground-truth formula -- there was
  no way to validate it against a real generation run since the SD3.5 weights
  aren't downloaded yet. Sanity-check it empirically once they are: the
  classifier-guided run should visibly steer *more* strongly at low
  ``sigma`` (near-clean latents late in sampling) and less at high ``sigma``
  (near-pure-noise, early), same qualitative shape as the DDIM version.

Also unlike SD1.x's single CLIP text encoder, SD3.5 has three (2x CLIP +
T5-XXL) and encode_prompt() returns both sequence embeddings *and* pooled
embeddings, both of which the transformer needs (``pooled_projections``) --
CFG-batching has to concat both, not just one.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from zennit.composites import NameMapComposite
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from ldce.sampling_helpers import cone_project, cone_project_chuncked, cone_project_chuncked_zero, _renormalize_gradient
from src.concept_cc_ddim import MaskHook, spatial_map, batch_map, _register_mask_fn


class ModifiedStableDiffusion3Img2ImgPipeline(StableDiffusion3Img2ImgPipeline):

    binary_classification_criterion = torch.nn.BCEWithLogitsLoss()

    def compute_pred_x0(self, sample, sigma, model_output):
        """Rectified-flow denoised-sample estimate: x0 = x_t - sigma * v_pred."""
        return sample - sigma * model_output

    def _sigma_for_timestep(self, t):
        """diffusers only exposes the current sigma via scheduler.step_index,
        which is lazily set the first time .step() runs -- we need it a step
        earlier (for the pred_x0/classifier-gradient computation, before
        .step() advances the sample), so this reaches into the same private
        init path .step() itself uses (`if self.step_index is None:
        self._init_step_index(timestep)`) rather than duplicating its index
        lookup logic."""
        if self.scheduler.step_index is None:
            self.scheduler._init_step_index(t)
        return self.scheduler.sigmas[self.scheduler.step_index]

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.6,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        # -- classifier-guidance / counterfactual steering, same names as
        # ModifiedStableDiffusionImg2ImgPipeline.__call__ so run_diffusers.py's
        # single pipe(...) call site works unchanged for both backbones.
        classifier=None,
        clf_transform=None,
        tgt=None,
        classifier_lambda=1.0,
        deg_cone_projection=45,
        lp_custom=1,
        dist_lambda=0.7,
        concept_conditioning=True,
        concept_conditions=None,
        spatial=False,
        uncondition_end=True,
        **kwargs,
    ):
        """Classifier-guided img2img counterfactual generation on SD3.5 -- see
        module docstring for the flow-matching guidance derivation. Same
        prompt/image/strength/... args as the stock
        StableDiffusion3Img2ImgPipeline.__call__, plus classifier/clf_transform/
        tgt/classifier_lambda/deg_cone_projection/lp_custom/dist_lambda/
        concept_conditioning/concept_conditions/spatial/uncondition_end for the
        counterfactual steering (same names as
        ModifiedStableDiffusionImg2ImgPipeline.__call__ in pipeline.py)."""
        callback_steps = kwargs.pop("callback_steps", None)
        _ = callback_steps  # accepted for call-site parity, unused (deprecated upstream too)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            strength,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=True,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        # classifier guidance here always needs both branches (score_out is
        # built from classifier/distance gradients, not blended CFG), unlike
        # stock SD3.5 which skips this when guidance_scale <= 1.
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        if latents is None:
            latents = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )

        distance_criterion = torch.nn.L1Loss(reduction='sum')

        # 6. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                orig_latents = latents.detach().clone()
                score_out = torch.zeros_like(latents)
                sigma = self._sigma_for_timestep(t)

                with torch.enable_grad():
                    x_noise = latents.detach().requires_grad_()

                    latent_model_input = torch.cat([x_noise] * 2)
                    timestep = t.expand(latent_model_input.shape[0])

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    pred_x0_latent = self.compute_pred_x0(x_noise, sigma, noise_pred_uncond)

                    pred_x0 = self.vae.decode(
                        (pred_x0_latent / self.vae.config.scaling_factor) + self.vae.config.shift_factor,
                        return_dict=False,
                    )[0]
                    pred_x0 = self.image_processor.postprocess(pred_x0, output_type='pt', do_denormalize=[True] * pred_x0.shape[0])
                    rpred_x0 = clf_transform(pred_x0)

                with torch.no_grad():
                    if lp_custom:
                        with torch.enable_grad():
                            from torchvision.transforms.v2.functional import resize
                            pred_x0_0to1 = resize(pred_x0, (256, 256), antialias=False).to(torch.float32)
                            image_resized = resize(image, (256, 256), antialias=False)

                            lp_dist = distance_criterion(pred_x0_0to1, image_resized.to(x_noise.device))
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=True)[0]

                    with torch.enable_grad():
                        if concept_conditioning:
                            hook_map = {}

                            if not uncondition_end or t.item() >= 50:
                                for key in concept_conditions.keys():
                                    if key not in hook_map:
                                        hook_map[key] = MaskHook([])

                                    if spatial:
                                        _register_mask_fn(hook_map[key], spatial_map, 0, concept_conditions[key], key)
                                    else:
                                        _register_mask_fn(hook_map[key], batch_map, 0, concept_conditions[key], key)

                            name_map = [([name], hook) for name, hook in hook_map.items()]
                            mask_composite = NameMapComposite(name_map)

                            with mask_composite.context(classifier) as modified:
                                pred_logits = modified(rpred_x0)
                        else:
                            pred_logits = classifier(rpred_x0)

                        y = tgt.to(device)

                        if len(pred_logits.shape) == 2:  # multi-class
                            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                            log_probs = log_probs[range(log_probs.size(0)), y.view(-1)]
                        else:  # binary
                            loss = self.binary_classification_criterion(pred_logits, y)
                            loss *= -1  # minimize this
                            log_probs = loss

                        grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]

                implicit_classifier_score = (noise_pred_cond - noise_pred_uncond)
                assert implicit_classifier_score.requires_grad == False

                cone_projection_type = "zero_binning"
                enforce_same_norms = True

                if classifier_lambda != 0:
                    # sigma replaces SD1.x's (1 - a_t).sqrt() -- see module
                    # docstring for the flow-matching derivation.
                    classifier_score = -1 * grad_classifier * sigma
                    assert classifier_score.requires_grad == False

                    projection_fn = cone_project if cone_projection_type == "default" else cone_project_chuncked
                    projection_fn = cone_project_chuncked_zero if "zero" in cone_projection_type else projection_fn

                    proj_out = projection_fn(
                        implicit_classifier_score.view(x_noise.shape[0], -1),
                        classifier_score.view(x_noise.shape[0], -1),
                        deg_cone_projection,
                        orig_shp=implicit_classifier_score.shape,
                    )

                    classifier_score = proj_out if cone_projection_type == "default" else proj_out[0].view_as(classifier_score)

                    if enforce_same_norms:
                        score_, norm_ = _renormalize_gradient(classifier_score, implicit_classifier_score)
                        classifier_score = classifier_lambda * score_
                    else:
                        classifier_score *= classifier_lambda

                    score_out += classifier_score

                if lp_custom:
                    lp_score = -1 * lp_grad * sigma

                    if enforce_same_norms:
                        score_, norm_ = _renormalize_gradient(lp_score, implicit_classifier_score)
                        lp_score = dist_lambda * score_
                    else:
                        lp_score *= dist_lambda

                    score_out -= lp_score

                noise_pred_final = noise_pred_uncond + self.guidance_scale * score_out

                latents = self.scheduler.step(noise_pred_final, t, orig_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
