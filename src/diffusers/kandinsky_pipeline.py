
from typing import Callable, Dict, List, Optional, Union

import PIL
import torch
from torchvision.transforms.functional import resize
from diffusers import Kandinsky3Img2ImgPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate, replace_example_docstring
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, EXAMPLE_DOC_STRING

from ldce.sampling_helpers import cone_project, cone_project_chuncked, cone_project_chuncked_zero, _renormalize_gradient

class ModifiedKandinskyImg2ImgPipeline(Kandinsky3Img2ImgPipeline):

    binary_classification_criterion = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
        strength: float = 0.3,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        classifier = None,
        tgt=None,
        classifier_lambda=1.,
        deg_cone_projection=45,
        lp_custom = 1,
        dist_lambda = 0.7,
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 3.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            attention_mask (`torch.Tensor`, *optional*):
                Pre-generated attention mask. Must provide if passing `prompt_embeds` directly.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Pre-generated negative attention mask. Must provide if passing `negative_prompt_embeds` directly.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`

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

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        cut_context = True
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            attention_mask,
            negative_attention_mask,
        )

        self._guidance_scale = guidance_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = self.encode_prompt(
            prompt,
            self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            _cut_context=cut_context,
            attention_mask=attention_mask,
            negative_attention_mask=negative_attention_mask,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            attention_mask = torch.cat([negative_attention_mask, attention_mask]).bool()
        # if not isinstance(image, list):
        #     image = [image]
        if not all(isinstance(i, (PIL.Image.Image, torch.Tensor)) for i in image):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support  PIL image and pytorch tensor"
            )

        print(f'image type {type(image)}')

        # image = torch.cat([prepare_image(i) for i in image], dim=0)
        image = image.to(dtype=prompt_embeds.dtype, device=device)
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        # 5. Prepare latents
        latents = self.movq.encode(image)["latents"]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents = self.prepare_latents(
            latents, latent_timestep, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )
        if hasattr(self, "text_encoder_offload_hook") and self.text_encoder_offload_hook is not None:
            self.text_encoder_offload_hook.offload()

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                orig_latents = latents.detach().clone()
                score_out = torch.zeros_like(latents)

                with torch.enable_grad():
                    x_noise = latents.detach().requires_grad_()

                    latent_model_input = torch.cat([x_noise] * 2) if self.do_classifier_free_guidance else latents

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=attention_mask,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

                        pred_x0_latent = self.compute_pred_x0(x_noise, t, noise_pred_uncond)

                    latents = self.scheduler.step(noise_pred_uncond, t, latents, **kwargs, return_dict=False)[0]
                    # pred_x0 = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                    # # pred_x0 = self.vae.decode(x_noise / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

                    # pred_x0 = self.image_processor.postprocess(pred_x0, output_type='pt', do_denormalize=[True] * pred_x0.shape[0])

                    pred_x0 = self.movq.decode(latents, force_not_quantize=True)["sample"]

                    # noise_pred = (guidance_scale + 1.0) * noise_pred_text - guidance_scale * noise_pred_uncond

                    # Modify here


                    distance_criterion = torch.nn.L1Loss(reduction='sum')

                    if lp_custom:          # changed from elif
                        rpred_x0 = resize(pred_x0, (256, 256)).to(torch.float32)
                        with torch.enable_grad():
                            pred_x0_0to1 = torch.clamp(rpred_x0, min=0.0, max=1.0)

                            lp_dist = distance_criterion(pred_x0_0to1, image.to(x_noise.device))
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=True)[0]

                    rpred_x0 = resize(pred_x0, (224, 224)).to(torch.float32)

                    pred_logits = classifier(rpred_x0)
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

                implicit_classifier_score = (noise_pred_text - noise_pred_uncond)
                # implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
                # check gradient tracking on implicit_classifier_score
                assert implicit_classifier_score.requires_grad == False, "implicit_classifier_score requires grad"


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

                noise_pred = guidance_scale * score_out + noise_pred_text
                # noise_pred = guidance_scale * (noise_pred_text - noise_pred_uncond) + noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    orig_latents,
                    generator=generator,
                ).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    attention_mask = callback_outputs.pop("attention_mask", attention_mask)
                    negative_attention_mask = callback_outputs.pop("negative_attention_mask", negative_attention_mask)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            # post-processing
            if output_type not in ["pt", "np", "pil", "latent"]:
                raise ValueError(
                    f"Only the output types `pt`, `pil`, `np` and `latent` are supported not output_type={output_type}"
                )
            if not output_type == "latent":
                image = self.movq.decode(latents, force_not_quantize=True)["sample"]

                if output_type in ["np", "pil"]:
                    print(f'image min {torch.min(image)}')
                    print(f'image max {torch.max(image)}')
                    # image = image * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                if output_type == "pil":
                    image = self.numpy_to_pil(image)
            else:
                image = latents

            self.maybe_free_model_hooks()

            if not return_dict:
                return (image,)

            return ImagePipelineOutput(images=image)

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
        
        return pred_original_sample
