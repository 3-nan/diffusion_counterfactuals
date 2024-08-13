import time
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torchvision import transforms
import zennit

from run_concept_ldce import get_classifier
from src.concept_cc_ddim import MaskHook, batch_map, spatial_map, _register_mask_fn

from DVCEs.blended_diffusion.utils_blended.metrics_accumulator import MetricsAccumulator
from DVCEs.blended_diffusion.utils_blended.model_normalization import ResizeAndMeanWrapper
from DVCEs.blended_diffusion.optimization.augmentations import ImageAugmentations
from DVCEs.blended_diffusion.optimization.dff_attack import (
    # DiffusionAttack, 
    CorrectedSummaryWriter,
    min_max_scale, 
    _map_img, 
    _renormalize_gradient, 
    compute_lp_dist,
    compute_lp_gradient,
    cone_projection,
    )
from DVCEs.blended_diffusion.guided_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)
from DVCEs.utils_svces.temperature_wrapper import TemperatureWrapper
from DVCEs.utils_svces.get_config import get_config
from DVCEs.utils_svces.Evaluator import Evaluator
# Define concept Diffusion Attack

# Run pipeline

class ConceptDiffusionAttack:

    def __init__(self, args, device) -> None:
        self.args = args
        self.probs = None
        self.y = None
        self.writer = None
        self.small_const = 1e-12
        self.tensorboard_counter = 0
        self.verbose = False    # args.verbose

        # if self.args.seed is not None:
        #     torch.manual_seed(self.args.seed)
        #     np.random.seed(self.args.seed)
        #     random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )


        self.device = device
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.num_classes = 1000
        self.model.load_state_dict(
            torch.load(
                f"checkpoints/256x256_diffusion_uncond.pt",
                map_location="cpu"
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)

        # if args.device_ids is not None and len(args.device_ids) > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=args.device_ids)

        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()

        if self.args.clip_guidance_lambda != 0:
            # if args.device_ids is not None and len(args.device_ids) > 1:
            #     self.model = nn.DataParallel(self.model, device_ids=args.device_ids)
            #     if self.args.clip_guidance_lambda != 0:
            #         self.clip_model = (
            #         nn.DataParallel(clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False), device_ids=args.device_ids)
            #     )
            # else:
            self.clip_model = (
                clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
            )
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()
        # args.device = device
        self.classifier_config = get_config(args)
        self.imagenet_labels = self.classifier_config.data.class_labels

        evaluator = Evaluator(args, self.classifier_config, {}, None, device=device)


        self.classifier = evaluator.load_model(
            self.args.classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_1, interpolation=self.args.interpolation_int_1)
        )
        print('temp o resize o mean wrapper on')
        self.classifier.to(self.device)
        self.classifier.eval()

        if self.args.second_classifier_type != -1:
            if self.args.second_classifier_type == -2:
                # self.second_classifier = TemperatureWrapper(ResizeAndMeanWrapper(get_classifier(args, device)))
                self.second_classifier = TemperatureWrapper(get_classifier(args, device))
            else:
                self.second_classifier = evaluator.load_model(
                    self.args.second_classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_2,
                                                                interpolation=self.args.interpolation_int_2)
                )

            self.second_classifier.to(self.device)
            self.second_classifier.eval()


        if self.args.third_classifier_type != -1:
            self.third_classifier = evaluator.load_model(
                self.args.third_classifier_type, prewrapper=partial(ResizeAndMeanWrapper, size=self.args.classifier_size_3,
                                                              interpolation=self.args.interpolation_int_3)
            )

            self.third_classifier.to(self.device)
            self.third_classifier.eval()

        ### ILVR resizers
        print("creating resizers...")

        down = lambda down_N : lambda tensor: resize_right.resize(tensor, 1 / down_N).to(self.device)
        up = lambda down_N : lambda tensor: resize_right.resize(tensor, down_N).to(self.device)
        self.resizers = (down, up)
        ### ILVR resizers


        self.clip_size = self.args.classifier_size_1

        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        if self.args.lpips_sim_lambda != 0:
            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)


        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        self.metrics = {**{"L"+str(p_): lambda x1, x2, p_=p_: (x1-x2).view(x1.shape[0], -1).norm(p=p_, dim=1) for p_ in [1, 2]}}


    def _compute_layers(self, x, classifier, model_name='resnet50'):

        if model_name.lower() == 'resnet50':
            return [classifier._modules[module_name](x) for module_name in classifier._modules.keys() if module_name not in ['fc', 'global_pool']]

    def _gram_schmidt(self, vv):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u

        print(vv.shape)
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[:, 0] = vv[:, 0].clone()
        for k in range(1, nk):
            vk = vv[:, k].clone()
            uk = 0
            for j in range(0, k):
                uj = uu[:, j].clone()
                uk = uk + projection(uj, vk)
            uu[:, k] = vk - uk
        for k in range(nk):
            uk = uu[:, k].clone()
            uu[:, k] = uk / uk.norm()
        return uu

    def _compute_probabilities(self, x, classifier,  permuted_logits_order=None):

        logits = classifier(_map_img(x))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if permuted_logits_order is not None:
            permuted_logits = torch.load('utils/'+permuted_logits_order)
            return log_probs[:, permuted_logits], probs[:, permuted_logits]
        else:
            return log_probs, probs

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def norms_per_image(self, x, y, tensor):
        return {str(self.imagenet_labels[y[i_val[0]]]): i_val[1].item() for i_val in
                enumerate(tensor)}

    def tensorboard_log_metrics(self, x, y, i):
        for name, metric in self.metrics.items():
            tensor = metric(x, self.init_image.add(1).div(2).clamp(0, 1))
            self.writer.add_scalars(f"Metrics/{name}", self.norms_per_image(x, y, tensor), i)

    def return_metrics_per_image(self, x):
        out = ['']*x.shape[0]

        for name, metric in self.metrics.items():
            tensor = metric(x, self.init_image.add(1).div(2).clamp(0, 1))
            out = [i_x[1]+f'{name}:{tensor[i_x[0]].item():.2f},' for i_x in enumerate(out)]
        return out

    def images_with_titles(self, imgs, titles):

        assert len(imgs) == len(titles)
        images = []
        for i, img in enumerate(imgs):
            fig = plt.figure(figsize=(14, 14))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.init_image[i].add(1).div(2).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy())
            images.append(plot_to_image(fig))

            fig = plt.figure(figsize=(14,14))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
            plt.title(titles[i], fontdict = {'fontsize' : 40})
            images.append(plot_to_image(fig))

        out = torch.cat(images, dim=0)
        print(out.shape)
        return out

    def clip_loss(self, x_in, text_embed):

        if self.mask is not None:
            masked_input = x_in
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2).clamp(0, 1)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        print('embeds shape', image_embeds.shape, text_embed.shape)
        dists = d_clip_loss(image_embeds, text_embed, use_cosine=True)

        print('clips dists are', dists.shape, dists)
        clip_loss = dists


        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed, use_cosine=True)

        return dists.item()

    def edit_image_by_prompt(self, x, y, dir, seed, conditions=None):


        # if not self.verbose:
        #     self.writer = EmptyWriter()
        # else:
        self.writer = CorrectedSummaryWriter(dir)



        self.text_batch = [self.imagenet_labels[y_el].split(',')[0] if y_el != -1 else 'flower' for y_el in y]

        if self.args.clip_guidance_lambda != 0:
            if y is not None:

                text_embeds = [self.clip_model.encode_text(
                    clip.tokenize(text_batch_).to(self.device)
                ).float() for text_batch_ in self.text_batch]
            else:
                self.text_batch = None
                text_embeds = self.clip_model.encode_text(
                   clip.tokenize(self.args.prompt).to(self.device)
                ).float()
            print('shape text_embed', len(text_embeds),  self.text_batch, len( self.text_batch))


        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        print('shapes x', x.shape[-1], self.model_config["image_size"])
        if x.shape[-1] != self.model_config["image_size"]:
            x = transforms.Resize(self.image_size)(x)
            print('shapes x after', x.shape)
        self.init_image = (x.to(self.device).mul(2).sub(1).clone().detach())

        loss_temp = torch.tensor(0.0).to(self.device)
        if self.args.second_classifier_type != -1:
            if self.args.projecting_cone:
                print('using cone projection, step0')
                logits = self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))
                logits2 = self.second_classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))
                if self.args.third_classifier_type != -1:
                    logits3 = self.third_classifier(
                        self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))

            else:
                print('using ensemble')
                logits = (0.5*self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1)).softmax(1) + 0.5*self.second_classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1)).softmax(1))
        else:
            logits = self.classifier(self.image_augmentations(self.init_image).add(1).div(2).clamp(0, 1))


        if self.args.second_classifier_type != -1 and not self.args.projecting_cone:
            log_probs = logits.log()
        else:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        current_bs = len(x)
        loss_indiv = log_probs[
            range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
        for i in range(current_bs):
            # We want to average at the "augmentations level"
            loss_temp += loss_indiv[i:: current_bs].mean()
        print('shape loss', loss_indiv.shape)
        print('targets', y.shape, y)

        if self.args.second_classifier_type != -1:
            if self.args.projecting_cone:
                self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]
                self.init_probs_second_classifier = logits2[:current_bs].softmax(1)[
                    range(current_bs), y.view(-1)]
                if self.args.third_classifier_type != -1:
                    self.init_probs_third_classifier = logits3[:current_bs].softmax(1)[
                        range(current_bs), y.view(-1)]
            else:
                self.probs = logits[:current_bs][range(current_bs), y.view(-1)]
        else:
            self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]

        self.init_probs = self.probs
        self.tensorboard_counter += 1



        self.mask = None


        def cond_fn_blended(x, t, y=None, eps=None, variance=None):
            print('use blended', self.args.classifier_lambda, self.args.background_preservation_loss, self.args.l2_sim_lambda, self.args.lpips_sim_lambda)
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)


                loss = torch.tensor(0)

                if self.args.clip_guidance_lambda != 0:
                    # ToDo: change text_embeds to incorporate only one image here
                    clip_loss = self.clip_loss(x_in, text_embeds) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())


                if self.args.classifier_lambda != 0:

                    loss_temp = torch.tensor(0.0).to(self.device)


                    logits = self.classifier(self.image_augmentations(x_in).add(1).div(2).clamp(0, 1))


                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                    loss_indiv = log_probs[
                        range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                    for i in range(current_bs):
                        # We want to average at the "augmentations level"
                        loss_temp += loss_indiv[i:: current_bs].mean()


                    self.probs = logits[:current_bs].softmax(1)[range(current_bs), y.view(-1)]


                    self.y = y
                    classifier_loss = loss_temp * self.args.classifier_lambda

                    loss = loss - classifier_loss


                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        masked_background = x_in
                    else:
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        loss = (
                            loss
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )



                self.tensorboard_counter += 1

                return -torch.autograd.grad(loss, x)[0]
        
        def concept_cond_fn_clean(x, t, y=None, eps=None, lp_custom=1.):

            current_bs = len(x)

            grad_out = torch.zeros_like(x)
            x = x.detach().requires_grad_()
            t = self.unscale_timestep(t)
            with torch.enable_grad():
                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )
                x_in = out["pred_xstart"]

            self.tensorboard_counter += 1

            # compute classifier gradient
            keep_denoising_graph = self.args.denoise_dist_input
            with torch.no_grad():

                if self.args.classifier_lambda != 0:
                    with torch.enable_grad():
                        # print('before classifier')

                        log_probs_1, probs_1 = self._compute_probabilities(self.image_augmentations(x_in), self.classifier)

                        target_log_confs_1 = log_probs_1[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]

                        # #########################################
                        # #   Add concept guidance here           #
                        # #########################################

                        # hook_map, y_targets = {}, []

                        # # # Free up last 50 generation steps
                        # # if not uncondition_end or t[0].item() >= 50:                               # TEST THIS FIRST
                        # #     for key in concept_conditions.keys():
                        # #         if key not in hook_map:
                        # #             hook_map[key] = MaskHook([])

                        # #         if spatial:
                        # #             _register_mask_fn(hook_map[key], spatial_map, 0, concept_conditions[key], key)
                        # #         else:
                        # #             _register_mask_fn(hook_map[key], batch_map, 0, concept_conditions[key], key)

                        # x_aug = _map_img(self.image_augmentations(x_in))

                        # name_map = [([name], hook) for name, hook in hook_map.items()]
                        # mask_composite = zennit.composites.NameMapComposite(name_map)

                        # with mask_composite.context(self.classifier) as modified:
                        # #     x = _map_img(pred_x0)
                        # #     if not self.classifier_wrapper: # only works for ImageNet!
                        # #         x = tf.center_crop(x, 224)
                        # #         x = normalize(x)

                        #     logits = modified(x_aug)

                        #     log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                        #     probs = torch.nn.functional.softmax(logits, dim=-1)

                        # target_log_confs_1 = log_probs[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]

                        grad_1 = torch.autograd.grad(target_log_confs_1.mean(), x, retain_graph=True)[0]
                        self.writer.add_images('gradients first classifier',
                                                min_max_scale(grad_1.abs().sum(1).unsqueeze(1)),
                                                self.tensorboard_counter
                                                )

                        if self.verbose:
                            print('maximizing prob_log', target_log_confs_1.shape, target_log_confs_1,
                                    probs_1)

                        if self.args.second_classifier_type != -1:

                            perturbed_images = self.image_augmentations(x_in)

                            if int(self.args.second_classifier_type) in [20]:
                                print("in 20")
                                grad_2 = 0
                                for i in range(int(self.args.aug_num)):
                                    print(i, perturbed_images.shape, perturbed_images[i].shape)
                                    log_probs_2, probs_2 = self._compute_probabilities(perturbed_images[i].unsqueeze(0), self.second_classifier,
                                                                                    permuted_logits_order=None)
                                    target_log_confs_2 = log_probs_2[range(current_bs), y.view(-1)]
                                    grad_2 += target_log_confs_2.numel() * \
                                                torch.autograd.grad(target_log_confs_2.mean(), x,
                                                                    retain_graph=keep_denoising_graph)[0]
                                    if self.verbose:
                                        print('second classifier probs_log', probs_2[range(current_bs), y.view(-1)])
                                grad_2 /= (target_log_confs_2.numel() * self.args.aug_num)
                            else:
                                self.writer.add_images('augmented images',
                                                        _map_img(perturbed_images),
                                                        self.tensorboard_counter
                                                        )
                                # log_probs_2, probs_2 = self._compute_probabilities(perturbed_images,
                                #                                                     self.second_classifier,
                                #                                                     permuted_logits_order=None)

                                #########################################
                                #   Add concept guidance here           #
                                #########################################

                                hook_map, y_targets = {}, []

                                # # Free up last 50 generation steps
                                if not self.args.optim or t[0].item() >= 50:                               # TEST THIS FIRST
                                    for key in conditions.keys():
                                        if key not in hook_map:
                                            hook_map[key] = MaskHook([])

                                        if self.args.spatial:
                                            _register_mask_fn(hook_map[key], spatial_map, 0, conditions[key], key)
                                        else:
                                            _register_mask_fn(hook_map[key], batch_map, 0, conditions[key], key)

                                x_aug = _map_img(self.image_augmentations(x_in))

                                name_map = [([name], hook) for name, hook in hook_map.items()]
                                mask_composite = zennit.composites.NameMapComposite(name_map)

                                with mask_composite.context(self.classifier) as modified:
                                #     x = _map_img(pred_x0)
                                #     if not self.classifier_wrapper: # only works for ImageNet!
                                #         x = tf.center_crop(x, 224)
                                #         x = normalize(x)

                                    logits2 = modified(x_aug)

                                    log_probs_2 = torch.nn.functional.log_softmax(logits2, dim=-1)
                                    probs_2 = torch.nn.functional.softmax(logits2, dim=-1)

                                    target_log_confs_2 = log_probs_2[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]


                                    if self.verbose:
                                        print('second classifier probs_log', probs_2[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)])
                                    # target_log_confs_2 = log_probs_2[range(current_bs * self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                                    grad_2 = \
                                        torch.autograd.grad(target_log_confs_2.mean(), x,
                                                            retain_graph=keep_denoising_graph)[0]

                            self.writer.add_images('gradients second classifier',
                                                    min_max_scale(grad_2.abs().sum(1).unsqueeze(1)),
                                                    self.tensorboard_counter
                                                    )
                            time_start = time.time()


                            # Cone Projection
                            grad_class = cone_projection(grad_1.view(x.shape[0], -1).cpu(),
                                                            grad_2.view(x.shape[0], -1).cpu(),
                                                            self.args.deg_cone_projection,
                                                            subspace_projection=False).view_as(grad_2).to(self.device)
                            print('projection_time', time.time() - time_start)
                            print('cone projection dist after', (grad_class - grad_2).norm(p=2))

                            # third classifier
                            if self.args.third_classifier_type != -1:

                                grad_class_third = cone_projection(grad_1.view(x.shape[0], -1),
                                                                (grad_3).view(x.shape[0], -1),
                                                                self.args.deg_cone_projection).view_as(grad_1)

                                grad_class -= grad_class_third
                        else:
                            grad_class = torch.autograd.grad(target_log_confs_1.mean(), x,
                                                                retain_graph=keep_denoising_graph)[0]

                    if self.args.enforce_same_norms:
                        grad_, norm_ = _renormalize_gradient(grad_class, eps)

                        grad_class = self.args.classifier_lambda * grad_

                    else:
                        grad_class *= self.args.classifier_lambda

                    grad_out += grad_class

                # distance gradients
                if lp_custom:
                    if not keep_denoising_graph:
                        print('not denoising_reguarization')
                        diff = x_in - self.init_image
                        lp_grad = compute_lp_gradient(diff, lp_custom)
                    else:
                        print('denoising_reguarization, new lpdist')
                        with torch.enable_grad():
                            lp_dist = compute_lp_dist(x_in, self.init_image, lp_custom)
                            lp_grad = torch.autograd.grad(lp_dist, x)[0]

                    if self.args.quantile_cut != 0:
                        pass

                    if self.args.enforce_same_norms:
                        print('enforcing same norms...')
                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)

                        lp_grad = self.args.lp_custom_value * grad_



                    else:
                        lp_grad *= self.args.lp_custom_value

                    grad_out -= lp_grad

                if self.args.layer_reg:
                    if not keep_denoising_graph:
                        diff = 0
                        for x_in_layer, init_image_layer in zip(self._compute_layers(x_in, self.classifier), self._compute_layers(self.init_image, self.classifier)):
                            diff += compute_lp_gradient(x_in_layer-init_image_layer, self.args.layer_reg)
                    else:
                        with torch.enable_grad():
                            diff = self._compute_layers(x_in, self.classifier) - self._compute_layers(self.init_image, self.classifier)
                            lp_dist = compute_lp_dist(diff, self.args.layer_reg)
                            lp_grad = torch.autograd.grad(lp_dist.mean(), x)[0]
                    if self.args.enforce_same_norms:
                        print('enforcing same norms...')

                        grad_, norm_ = _renormalize_gradient(lp_grad, eps)

                        lp_grad = self.args.layer_reg_value * grad_

                    else:
                        lp_grad *= self.args.layer_reg_value

                    grad_out -= lp_grad

            return grad_out

        @torch.no_grad()
        def postprocess_fn(out, t):

            if self.mask is not None:

                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])

                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)


            return out

        targets_classifier = y

        if self.args.gen_type == 'ddim':
            gen_func = self.diffusion.ddim_sample_loop_progressive
        elif self.args.gen_type == 'p_sample':
            gen_func = self.diffusion.p_sample_loop_progressive
        else:
            raise ValueError(f'Generation type {self.args.gen_type} is not implemented.')

        samples = gen_func(
            self.model,
            (
                current_bs, # * len(self.args.device_ids),
                3,
                self.model_config["image_size"],
                self.model_config["image_size"],
            ),
            clip_denoised=False,
            model_kwargs={

                "y": torch.tensor(targets_classifier, device=self.device, dtype=torch.long)
            },
            cond_fn=cond_fn_blended if self.args.use_blended else concept_cond_fn_clean,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=self.init_image if not self.args.not_use_init_image else None,
            postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
            randomize_class=False,
            resizers=self.resizers,
            range_t=self.args.range_t,
            eps_project=self.args.eps_project,
            ilvr_multi=self.args.ilvr_multi,
            seed=seed

        )

        total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
        print('num total steps is', total_steps)
        max_probs = self.init_probs * 0
        sample_final = self.init_image
        print('before loop')
        for i, sample in enumerate(samples):
            print(i, max_probs)

            if i == total_steps:
                sample_final = sample["pred_xstart"]


        self.writer.flush()
        self.writer.close()
        return _map_img(sample_final).clamp(0, 1)

    def perturb(self, x, y, dir, seed, conditions=None):

        # torch.manual_seed(self.args.seed)
        # random.seed(self.args.seed)
        # np.random.seed(self.args.seed)

        self.tensorboard_counter = 0
        adv_best = self.edit_image_by_prompt(x, y, dir, seed, conditions=conditions)
        return [adv_best]