""" Extended cc_ddim script from 
    https://github.com/lmb-freiburg/ldce/blob/main/ldm/models/diffusion/cc_ddim.py
    used CRP from https://github.com/rachtibat/zennit-crp/blob/master/crp/hooks.py
"""
from typing import Any, Dict, List
import weakref
import functools
import numpy as np
import torch
import torchvision.transforms.functional as tf
from zennit.composites import NameMapComposite
from zennit.core import Hook, RemovableHandle, RemovableHandleList
from ldce.ldm.models.diffusion.cc_ddim import CCMDDIMSampler
from ldce.sampling_helpers import cone_project, cone_project_chuncked, cone_project_chuncked_zero, normalize, _renormalize_gradient, _map_img


class MaskHook(Hook):
    """ Hook for masking on specified channels in the gradient computation. """

    def __init__(self, fn_list):

        self.fn_list = fn_list

    def post_forward(self, module, input, output):
        '''Register a backward-hook to the resulting tensor right after the forward.'''
        hook_ref = weakref.ref(self)

        @functools.wraps(self.backward)
        def wrapper(grad):
            return hook_ref().backward(module, grad)

        if not isinstance(output, tuple):
            output = (output,)

        if output[0].grad_fn is not None:
            # only if gradient required
            output[0].register_hook(wrapper)
        return output[0] if len(output) == 1 else output
    
    def backward(self, module, grad):
        '''Hook applied during backward-pass'''
        for mask_fn in self.fn_list:
            grad = mask_fn(grad)

        return grad
    
    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same fn_list list.
        '''
        return self.__class__(fn_list=self.fn_list)

    def remove(self):
        '''When removing hooks, remove all stored mask_fn.'''
        self.fn_list.clear()

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])

    # def backward(self, module, grad_input, grad_output):
    #     """ Modify gradient in the backward pass. """

    #     gradient = grad_input[0]

    #     mask = torch.zeros_like(gradient)

    #     return (gradient * mask, )
# @staticmethod
def mask_map(batch_id: int, concept_ids: List, layer_name=None):
    """
    Wrapper that generates a function that modifies the gradient (replaced by zennit by attributions).

    Parameters:
    ----------
    batch_id: int
        Specifies the batch dimension in the torch.Tensor.
    concept_ids: list of integer values
        integer lists corresponding to channel indices.

    Returns:
    --------
    callable function that modifies the gradient
    """

    def mask_fct(grad):

        mask = torch.zeros_like(grad[batch_id])
        mask[concept_ids] = 1
        grad[batch_id] = grad[batch_id] * mask

        return grad

    return mask_fct

def _register_mask_fn(hook, mask_map, b_index, c_indices, l_name):

    if callable(mask_map):
        mask_fn = mask_map(b_index, c_indices, l_name)
    elif isinstance(mask_map, Dict):
        mask_fn = mask_map[l_name](b_index, c_indices, l_name)
    else:
        raise ValueError("<mask_map> must be a dictionary or callable function.")

    hook.fn_list.append(mask_fn)

class GradConditioner:

    def __init__(self, model, device=None, overwrite_data_grad=True, no_param_grad=True) -> None:

        self.MODEL_OUTPUT_NAME = "y"

        self.device = next(model.parameters()).device if device is None else device
        self.model = model
        self.overwrite_data_grad = overwrite_data_grad

        if no_param_grad:
            self.model.requires_grad_(False)
    
    def backward(self, pred, grad_mask, partial_backward, layer_names, layer_out, generate=False):

        if partial_backward and len(layer_names) > 0:

            wrt_tensor, grad_tensors = pred, grad_mask.to(pred)

            for l_name in layer_names:

                inputs = layer_out[l_name]

                try:
                    grad = torch.autograd.grad(wrt_tensor, inputs=inputs, grad_outputs=grad_tensors, retain_graph=True)
                except RuntimeError as e:
                    if "allow_unused=True" not in str(e):
                        raise e
                    else:
                        raise RuntimeError(
                            "The layer names must be ordered according to their succession in the model if 'exclude_parallel'=True."
                            " Please make sure to start with the last and end with the first layer in each condition dict. In addition,"
                            " parallel layers can not be used in one condition.")

                # TODO: necessary?
                if grad is None:
                    raise RuntimeError(
                        "The layer names must be ordered according to their succession in the model if 'exclude_parallel'=True."
                        " Please make sure to start with the last and end with the first layer in each condition dict. In addition,"
                        " parallel layers can not be used in one condition.")

                wrt_tensor, grad_tensors = layer_out[l_name], grad

            torch.autograd.backward(wrt_tensor, grad_tensors, retain_graph=generate)

        else:

            torch.autograd.backward(pred, grad_mask.to(pred), retain_graph=generate)


    def __call__(self, data, conditions, mask_map, init_rel=None, exclude_parallel=False) -> torch.Tensor:

        hook_map, y_targets, cond_l_names = {}, [], []
        for i, cond in enumerate(conditions):
            for l_name, indices in cond.items():
                if l_name == self.MODEL_OUTPUT_NAME:
                    y_targets.append(indices)
                else:
                    if l_name not in hook_map:
                        hook_map[l_name] = MaskHook([])
                    self._register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)
                    if l_name not in cond_l_names:
                        cond_l_names.append(l_name)

        name_map = [([name], hook) for name, hook in hook_map.items()]
        mask_composite = NameMapComposite(name_map)

        with mask_composite.context(self.model) as modified:

            pred = modified(data)
            grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
            # self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)
            self.backward(pred, grad_mask, exclude_parallel, None, None)

        gradient = data.grad.detach()

        return gradient


class ConceptCCMDDIMSampler(CCMDDIMSampler):
    """ Extended implementation of the CCMDDIM Sampler.
        Concept guidance shall be added.
    """

    def conditional_score(self, x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale=1, unconditional_conditioning=None, y=None):
        # return super().conditional_score(x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale, unconditional_conditioning, y)
        """
        Args:
            x: input image
            t: time step
            c: conditioning
            index: index for the schedule
            use_original_steps: whether to use the original steps
            quantize_denoised: whether to quantize the denoised image
            unconditional_guidance_scale: scale for the unconditional guidance
            unconditional_conditioning: unconditional conditioning
            y: target class


        Returns:
            e_t: score after conditioning

        """
        b, *_, device = *x.shape, x.device
        x = x.detach()  # .requires_grad_()
        # x.requires_grad = True
        prob_best_class = None
        mask_guidance = None

        ## check if gradient tracking is on for x
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        # if self.guidance == "free":
        #     e_t_uncond, e_t, pred_x0 = self.get_output(x, t, c, index, unconditional_conditioning, use_original_steps,
        #                                                quantize_denoised, return_decoded=True)

        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        #     return e_t

        # print("check gradient tracking onf e ", e_t.requires_grad)
        score_out = torch.zeros_like(x)

        # Get output of the model -> e_t_uncond, e_t, pred_x0
        with torch.enable_grad():
            x_noise = x.detach().requires_grad_()
            ret_vals = self.get_output(x_noise, t, c, index, unconditional_conditioning,
                                                        use_original_steps, quantize_denoised=quantize_denoised,
                                                        return_decoded=True, return_pred_latent_x0=self.log_backprop_gradients)
            if self.log_backprop_gradients:
                e_t_uncond, e_t, pred_x0, pred_latent_x0 = ret_vals
            else:
                e_t_uncond, e_t, pred_x0 = ret_vals

        with torch.no_grad():
            # if isinstance(self.lp_custom, str) and "dino_" in self.lp_custom: # retain_graph causes cuda oom issues for dino distance regularizer...
            #     with torch.enable_grad():
            #         pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)
            #         lp_dist = self.distance_criterion(pred_x0_0to1, self.dino_init_features.to(x.device).detach())
            #         lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=False)[0]

            if self.lp_custom:          # changed from elif
                with torch.enable_grad():
                    pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)
                    lp_dist = self.distance_criterion(pred_x0_0to1, self.init_images.to(x.device))
                    lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=True)[0]

                    print(f"Init_images: {self.init_images.size()}")
                    print(f"lp_grad: {lp_grad.size()}, lp_dist: {lp_dist}, pred_x0_0to1: {pred_x0_0to1.size()}")

            #########################################
            # Get classifier prediction & gradients
            #########################################
            if self.classifier_lambda != 0:
                with torch.enable_grad():
                    # if isinstance(self.lp_custom, str) and "dino_" in self.lp_custom:
                    #     x_noise = x.detach().requires_grad_()
                    #     ret_vals = self.get_output(x_noise, t, c, index, unconditional_conditioning,
                    #                                                 use_original_steps, quantize_denoised=quantize_denoised,
                    #                                                 return_decoded=True, return_pred_latent_x0=self.log_backprop_gradients)
                    #     if self.log_backprop_gradients:
                    #         e_t_uncond, e_t, pred_x0, pred_latent_x0 = ret_vals
                    #     else:
                    #         e_t_uncond, e_t, pred_x0 = ret_vals
                    pred_logits = self.get_classifier_logits(pred_x0)
                    print(f"classifier logits: {pred_logits.size()} for pred_x0: {pred_x0.size()}")
                    if len(pred_logits.shape) == 2: # multi-class
                        log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
                        log_probs = log_probs[range(log_probs.size(0)), y.view(-1)]
                        prob_best_class = torch.exp(log_probs).detach()
                    else: # binary
                        loss = self.binary_classification_criterion(pred_logits, y)
                        loss *= -1 # minimize this
                        log_probs = loss
                        prob_best_class = pred_logits.sigmoid().detach()

                    if self.log_backprop_gradients: pred_latent_x0.retain_grad()

                    # if self.dino_pipeline:
                    #     grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=False)[0]
                    # else:
                    grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]
                        # grad_classifier2 = torch.autograd.grad(log_probs[0].sum(), x_noise, retain_graph=False)[0]

                    print(f"Log probs shape: {log_probs.size()}, x_noise: {x_noise.size()}")
                    print(f"Grad classifier shape: {grad_classifier.size()}")

                    if self.log_backprop_gradients:
                        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
                        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
                        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                        a_t_sqrt = a_t.sqrt()
                        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)
                        grad_pred_latent_x0 = pred_latent_x0.grad.data
                        grad_unet_wrt_zt = (grad_classifier*a_t_sqrt/grad_pred_latent_x0 - 1)*(-1/sqrt_one_minus_at)

                        cossim = torch.nn.CosineSimilarity()
                        cossim_wpre = cossim(grad_classifier.view(2, -1), grad_pred_latent_x0.view(2, -1))
                        
                        print(torch.norm(grad_classifier, dim=(2,3)), torch.norm(grad_pred_latent_x0, dim=(2,3)), torch.norm(grad_unet_wrt_zt, dim=(2,3)))
                        print(cossim_wpre)

        # assert e_t_uncond.requires_grad == True and e_t.requires_grad == True, "e_t_uncond and e_t should require gradients"

        # if self.guidance == "projected":
        implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
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
            
            print(f"Proj out shape {proj_out[0].size()}")

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

        # distance gradients
        if self.lp_custom:

            lp_score = -1 * lp_grad * (1 - a_t).sqrt()

            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(lp_score,
                                                      implicit_classifier_score)
                lp_score = self.dist_lambda * score_

            else:

                lp_score *= self.dist_lambda

            score_out -= lp_score

        e_t = e_t_uncond + unconditional_guidance_scale * score_out  # (1 - a_t).sqrt() * grad_out

        print(f"Score out: {score_out.size()}, e_t: {e_t.size()}")


        if self.record_intermediate_results:
            # adding images to create a gif
            pred_x0_copy = pred_x0.clone().detach()
            img = torch.clamp(_map_img(pred_x0_copy), min=0.0, max=1.0)
            #img = torch.permute(img, (1, 2, 0, 3)).reshape((img.shape[1], img.shape[2], -1))

            self.images.append(img.detach().cpu())
            if self.classifier_lambda != 0 and self.cone_projection_type == "binning":
                self.concensus_regions.append(concensus_region.detach().cpu())

            if prob_best_class is not None:
                self.probs.append(prob_best_class.detach().cpu())

        return e_t
    

    # concept_conditional_score
    def new_conditional_score(self, x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale=1, unconditional_conditioning=None, y=None):
        # return super().conditional_score(x, t, c, index, use_original_steps, quantize_denoised, unconditional_guidance_scale, unconditional_conditioning, y)
        """
        Args:
            x: input image
            t: time step
            c: conditioning
            index: index for the schedule
            use_original_steps: whether to use the original steps
            quantize_denoised: whether to quantize the denoised image
            unconditional_guidance_scale: scale for the unconditional guidance
            unconditional_conditioning: unconditional conditioning
            y: target class


        Returns:
            e_t: score after conditioning

        """

        b, *_, device = *x.shape, x.device
        x = x.detach()
        # prob_best_class = None
        mask_guidance = None

        ## check if gradient tracking is on for x
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
            return e_t
        
        score_out = torch.zeros_like(x)

        # Get output of denoising model
        with torch.enable_grad():
            x_noise = x.detach().requires_grad_()
            ret_vals = self.get_output(x_noise, t, c, index, unconditional_conditioning,
                                                        use_original_steps, quantize_denoised=quantize_denoised,
                                                        return_decoded=True, return_pred_latent_x0=self.log_backprop_gradients)
            if self.log_backprop_gradients:
                e_t_uncond, e_t, pred_x0, pred_latent_x0 = ret_vals
            else:
                e_t_uncond, e_t, pred_x0 = ret_vals

        with torch.no_grad():
            if self.lp_custom:          # changed from elif
                with torch.enable_grad():
                    pred_x0_0to1 = torch.clamp(_map_img(pred_x0), min=0.0, max=1.0)
                    lp_dist = self.distance_criterion(pred_x0_0to1, self.init_images.to(x.device))
                    lp_grad = torch.autograd.grad(lp_dist.mean(), x_noise, retain_graph=True)[0]

                    # print(f"Init_images: {self.init_images.size()}")
                    # print(f"lp_grad: {lp_grad.size()}, lp_dist: {lp_dist}, pred_x0_0to1: {pred_x0_0to1.size()}")
            #########################################
            # Get classifier prediction & gradients
            #########################################
            if self.classifier_lambda != 0:

                # add conditions for masking
                conditions = [{'layer3.5.conv1': range(40)}]

                with torch.enable_grad():

                    hook_map, y_targets = {}, []
                    for i, cond in enumerate(conditions):
                        for l_name, indices in cond.items():
                            # if l_name == self.MODEL_OUTPUT_NAME:
                            if l_name == 'y':
                                y_targets.append(indices)
                            else:
                                if l_name not in hook_map:
                                    hook_map[l_name] = MaskHook([])
                                _register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)

                    name_map = [([name], hook) for name, hook in hook_map.items()]
                    mask_composite = NameMapComposite(name_map)

                    with mask_composite.context(self.classifier) as modified:
                        x = _map_img(pred_x0)
                        if not self.classifier_wrapper: # only works for ImageNet!
                            x = tf.center_crop(x, 224)
                            x = normalize(x)
                        pred_logits = modified(x)

                    # pred_logits = self.get_classifier_logits(pred_x0)
                        # print(f"classifier logits: {pred_logits.size()} for pred_x0: {pred_x0.size()}")
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

                        if self.log_backprop_gradients: pred_latent_x0.retain_grad()

                    #############################################################################
                    # Concept-based conditioning                                                #
                    # In the backward pass of the classifier, the gradient computation          #
                    # can be constrained to specific concepts.                                  #
                    # Thereby, the gradient w.r.t pred_x0 changes, which is backpropagated      #
                    # through the First stage decoder to compute the gradient w.r.t x_noise.    #
                    # By constraining the gradient to single concepts, we hope to restrain      #
                    # the generation to class-specific changes in these concepts only.          #
                    #############################################################################
                        # print([m for m, n in self.classifier.named_modules()])

                        grad_classifier = torch.autograd.grad(log_probs.sum(), x_noise, retain_graph=True)[0]
                    # grad_classifier2 = torch.autograd.grad(log_probs[0].sum(), x_noise, retain_graph=False)[0]

                        # print(f"Log probs shape: {log_probs.size()}, x_noise: {x_noise.size()}")
                        # print(f"Grad classifier shape: {grad_classifier.size()}")

        implicit_classifier_score = (e_t - e_t_uncond)  # .detach()
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
            
            # print(f"Proj out shape {proj_out[0].size()} with min {torch.min(proj_out[0])} and max {torch.max(proj_out[0])}")

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

        # distance gradients
        if self.lp_custom:

            lp_score = -1 * lp_grad * (1 - a_t).sqrt()

            if self.enforce_same_norms:
                score_, norm_ = _renormalize_gradient(lp_score,
                                                      implicit_classifier_score)
                lp_score = self.dist_lambda * score_

            else:

                lp_score *= self.dist_lambda

            score_out -= lp_score

        e_t = e_t_uncond + unconditional_guidance_scale * score_out  # (1 - a_t).sqrt() * grad_out

        # print(f"Score out: {score_out.size()}, e_t: {e_t.size()}")


        # if self.record_intermediate_results:
        #     # adding images to create a gif
        #     pred_x0_copy = pred_x0.clone().detach()
        #     img = torch.clamp(_map_img(pred_x0_copy), min=0.0, max=1.0)
        #     #img = torch.permute(img, (1, 2, 0, 3)).reshape((img.shape[1], img.shape[2], -1))

        #     self.images.append(img.detach().cpu())
        #     if self.classifier_lambda != 0 and self.cone_projection_type == "binning":
        #         self.concensus_regions.append(concensus_region.detach().cpu())

        #     if prob_best_class is not None:
        #         self.probs.append(prob_best_class.detach().cpu())

        return e_t

