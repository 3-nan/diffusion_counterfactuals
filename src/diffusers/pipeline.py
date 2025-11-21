
import os
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torchvision.transforms.v2.functional import center_crop, resize
import requests
from io import BytesIO
from PIL import Image
from zennit.composites import NameMapComposite
from diffusers import AutoPipelineForImage2Image, StableDiffusionImg2ImgPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate, replace_example_docstring
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, EXAMPLE_DOC_STRING

from ldce.sampling_helpers import cone_project, cone_project_chuncked, cone_project_chuncked_zero, _renormalize_gradient
from src.concept_cc_ddim import MaskHook, spatial_map, batch_map, _register_mask_fn

class ModifiedStableDiffusionImg2ImgPipeline(StableDiffusionImg2ImgPipeline):

    binary_classification_criterion = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        classifier = None,
        clf_transform = None,
        tgt=None,
        classifier_lambda=1.,
        deg_cone_projection=45,
        lp_custom = 1,
        dist_lambda = 0.7,
        concept_conditioning=True,
        concept_conditions = None,
        spatial=False,
        uncondition_end=True,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                orig_latents = latents.detach().clone()
                score_out = torch.zeros_like(latents)

                with torch.enable_grad():
                    x_noise = latents.detach().requires_grad_()

                    # expand the latents if we are doing classifier free guidance
                    # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = torch.cat([x_noise] * 2) if self.do_classifier_free_guidance else x_noise
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                    # x_noise = latent_model_input.detach().requires_grad_()

                    noise_pred = self.unet(
                        latent_model_input, #x_noise,        # latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    pred_x0_latent = self.compute_pred_x0(x_noise, t, noise_pred_uncond)

                    # latentsstep = self.scheduler.step(noise_pred_uncond, t, latents, **extra_step_kwargs, return_dict=True) #[0]
                    # latents = latentsstep.prev_sample
                    # pred_x0_latent = latentsstep.pred_original_sample
    
                    # with torch.enable_grad():
                    pred_x0 = self.vae.decode(pred_x0_latent / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                    # pred_x0 = self.vae.decode(x_noise / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

                    pred_x0 = self.image_processor.postprocess(pred_x0, output_type='pt', do_denormalize=[True] * pred_x0.shape[0])

                    # rpred_x0 = center_crop(resize(pred_x0, (256, 256), antialias=True), (224, 224)).to(torch.float32)
                    rpred_x0 = clf_transform(pred_x0)

                with torch.no_grad():

                    # lp_custom = 0
                    distance_criterion = torch.nn.L1Loss(reduction='sum')

                    if lp_custom:          # changed from elif
                        with torch.enable_grad():
                            pred_x0_0to1 = resize(pred_x0, (256, 256), antialias=False).to(torch.float32)
                            # pred_x0_0to1 = resize(pred_x0, (512, 512)).to(torch.float32)
                            # pred_x0_0to1 = torch.clamp(rpred_x0, min=0.0, max=1.0)
                            image_resized = resize(image, (256, 256), antialias=False)

                            lp_dist = distance_criterion(pred_x0_0to1, image_resized.to(x_noise.device))
                            # print(lp_dist.mean().device)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=True)[0]


                    with torch.enable_grad():
                        if concept_conditioning:
                            hook_map, y_targets = {}, []

                        # print(f'timestep: {t.item()}')
                    # # Free up last 50 generation steps
                            if not uncondition_end or t.item() >= 50:                               # TEST THIS FIRST
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
                                # x = _map_img(pred_x0)
                                # if not self.classifier_wrapper: # only works for ImageNet!
                                #     x = tf.center_crop(x, 224)
                                #     x = normalize(x)

                                pred_logits = modified(rpred_x0)

                        else:
                            pred_logits = classifier(rpred_x0)
                    # pred_logits = classifier(rpred_x0)
                    # pred_logits = self.get_classifier_logits(pred_x0)
                    # print(pred_logits)

                        y=tgt.to(device)

                        if len(pred_logits.shape) == 2: # multi-class
                            # print("Multiclass model used. ")
                            log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                            log_probs = log_probs[range(log_probs.size(0)), y.view(-1)]
                            prob_best_class = torch.exp(log_probs).detach()
                        else: # binary
                            loss = self.binary_classification_criterion(pred_logits, y)
                            loss *= -1 # minimize this
                            log_probs = loss
                            prob_best_class = pred_logits.sigmoid().detach()

                        grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]

                
                implicit_classifier_score = (noise_pred_cond - noise_pred_uncond)
                # implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
                # check gradient tracking on implicit_classifier_score
                assert implicit_classifier_score.requires_grad == False, "implicit_classifier_score requires grad"

                # lp_custom = 1
                # dist_lambda = 0.3
                use_original_steps = True
                cone_projection_type = "zero_binning"
                enforce_same_norms = True # False

                if lp_custom or classifier_lambda != 0:
                    b = x_noise.shape[0]
                    alphas = self.scheduler.alphas_cumprod if use_original_steps else self.ddim_alphas
                    a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)

                if classifier_lambda != 0:
                    classifier_score = -1 * grad_classifier * (1 - a_t).sqrt()              # scaled target model gradients
                    # print(f'classifier score 1 {classifier_score.size()}')
                    assert classifier_score.requires_grad == False, "classifier_score requires grad"


                    # project the gradient of the classifier on the implicit classifier
                    projection_fn = cone_project if cone_projection_type == "default" else cone_project_chuncked
                    projection_fn = cone_project_chuncked_zero if "zero" in cone_projection_type else projection_fn

                    # projection function: zero_binning -> cone_project_chuncked_zero
                    proj_out = projection_fn(implicit_classifier_score.view(x_noise.shape[0], -1),
                                                    classifier_score.view(x_noise.shape[0], -1),
                                                    deg_cone_projection,
                                                    orig_shp=implicit_classifier_score.shape) #\
                        # if self.guidance == "projected" else classifier_score
                    # print(f'proj_out score {proj_out[0].size()}')

                    classifier_score = proj_out if cone_projection_type == "default" else proj_out[0].view_as(classifier_score)
                    concensus_region = proj_out[1].unsqueeze(1) if cone_projection_type == "binning" else None

                    if enforce_same_norms:
                        # print(f'classifier score {classifier_score.size()}')
                        # print(f'implicit score {implicit_classifier_score.size()}')
                        score_, norm_ = _renormalize_gradient(classifier_score,
                                                            implicit_classifier_score)  # e_t_uncond (AWAREE!!)
                        classifier_score = classifier_lambda * score_

                    else:
                        classifier_score *= classifier_lambda

                    score_out += classifier_score

                # distance gradients
                if lp_custom:

                    lp_score = -1 * lp_grad * (1 - a_t).sqrt()

                    if enforce_same_norms:
                        score_, norm_ = _renormalize_gradient(lp_score,
                                                            implicit_classifier_score)
                        lp_score = dist_lambda * score_

                    else:

                        lp_score *= dist_lambda

                    score_out -= lp_score

                # noise_pred = noise_pred_uncond + self.guidance_scale * score_out

                noise_pred = noise_pred_uncond + self.guidance_scale * implicit_classifier_score # (noise_pred_cond - noise_pred_uncond)

                    # noise_pred = self.perform_conditioning(noise_pred_uncond, noise_pred_cond, latents, x_noise, generator, classifier, y=tgt.to(device))

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, orig_latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def perform_conditioning(self, noise_pred_uncond, noise_pred_cond, latents, x_noise, generator, classifier, y=None):
        """ Include classifier conditioning. """
        # get pred_x0
        # alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        # sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        # sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
        # # current prediction for x_0; get the original image with range [0, 1] if it is in latent space
        # pred_latent_x0 = (x - sqrt_one_minus_at * e_t_uncond) / a_t.sqrt()  # e_t - >  e_t_uncond
        # if quantize_denoised:
        #     pred_latent_x0, _, *_ = self.model.first_stage_model.quantize(pred_latent_x0)

        # pred_x0 = self.model.differentiable_decode_first_stage(
        #     pred_latent_x0)

        with torch.enable_grad():
            pred_x0 = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

            pred_x0 = self.image_processor.postprocess(pred_x0, output_type='pt', do_denormalize=[True] * pred_x0.shape[0])

            print(torch.max(pred_x0))
            print(torch.min(pred_x0))

            pred_logits = classifier(pred_x0)
            # pred_logits = self.get_classifier_logits(pred_x0)
            print(pred_logits)

        # hook_map, y_targets = {}, []

        # # Free up last 50 generation steps
        # if not uncondition_end or t[0].item() >= 50:
        #     for key in concept_conditions.keys():
        #         if key not in hook_map:
        #             hook_map[key] = MaskHook([])

        #         if spatial:
        #             _register_mask_fn(hook_map[key], spatial_map, 0, concept_conditions[key], key)
        #         else:
        #             _register_mask_fn(hook_map[key], batch_map, 0, concept_conditions[key], key)

        # name_map = [([name], hook) for name, hook in hook_map.items()]
        # mask_composite = NameMapComposite(name_map)

        # with mask_composite.context(self.classifier) as modified:
        #     x = _map_img(pred_x0)
        #     if not self.classifier_wrapper: # only works for ImageNet!
        #         x = tf.center_crop(x, 224)
        #         x = normalize(x)

        #     pred_logits = modified(x)

            if len(pred_logits.shape) == 2: # multi-class
                # print("Multiclass model used. ")
                log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                log_probs = log_probs[range(log_probs.size(0)), y.view(-1)]
                prob_best_class = torch.exp(log_probs).detach()
            else: # binary
                loss = self.binary_classification_criterion(pred_logits, y)
                loss *= -1 # minimize this
                log_probs = loss
                prob_best_class = pred_logits.sigmoid().detach()

            grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]
        # grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]


        implicit_classifier_score = (noise_pred_cond - noise_pred_uncond)
        # implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
        # check gradient tracking on implicit_classifier_score
        assert implicit_classifier_score.requires_grad == False, "implicit_classifier_score requires grad"

        if self.lp_custom or self.classifier_lambda != 0:
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)

        if self.classifier_lambda != 0:
            classifier_score = -1 * grad_classifier * (1 - a_t).sqrt()              # scaled target model gradients
            assert classifier_score.requires_grad == False, "classifier_score requires grad"

            # project the gradient of the classifier on the implicit classifier
            projection_fn = cone_project if self.cone_projection_type == "default" else cone_project_chuncked
            projection_fn = cone_project_chuncked_zero if "zero" in self.cone_projection_type else projection_fn


            # projection function: zero_binning -> cone_project_chuncked_zero
            proj_out = projection_fn(implicit_classifier_score.view(x.shape[0], -1),
                                            classifier_score.view(x.shape[0], -1),
                                            self.deg_cone_projection,
                                            orig_shp=implicit_classifier_score.shape) \
                if self.guidance == "projected" else classifier_score

            classifier_score = proj_out if self.cone_projection_type == "default" else proj_out[0].view_as(classifier_score)
            concensus_region = proj_out[1].unsqueeze(1) if self.cone_projection_type == "binning" else None
            #print(classifier_score.shape, concensus_region.shape)
            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(classifier_score,
                                                      implicit_classifier_score)  # e_t_uncond (AWAREE!!)
                classifier_score = self.classifier_lambda * score_

            else:
                classifier_score *= self.classifier_lambda

            score_out += classifier_score


            noise_pred = noise_pred_uncond + self.guidance_scale * score_out

            return noise_pred

        
    def compute_pred_x0(self, sample, t, model_output):

        # t = timestep
        # prev_t = self.scheduler.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        
        # print(self.scheduler.config)
        # print(type(self.scheduler))

        # 3. Clip or threshold "predicted x_0"
        # if self.scheduler.config.thresholding:
        #     pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
        # elif self.scheduler.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
        #     )
        
        return pred_original_sample

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_id_or_path = "runwayml/stable-diffusion-v1-5"

# # pipe = AutoPipelineForImage2Image.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = ModifiedStableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(device)

# # classifier_model = get_classifier(cfg, device)
# # classifier_model.to(device).eval()

# prompt = "A fantasy landscape, Cinematic lighting"
# negative_prompt = "low quality, bad quality"

# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
 
# response = requests.get(url)
# original_image = Image.open(BytesIO(response.content)).convert("RGB")
# original_image.thumbnail((768, 768))

# # response = requests.get(url)
# # init_image = Image.open(BytesIO(response.content)).convert("RGB")
# # init_image = init_image.resize((768, 512))

# image = pipe(prompt=prompt, image=original_image, strength=0.3).images[0]
# # images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# image.save("fantasy_land.png")
