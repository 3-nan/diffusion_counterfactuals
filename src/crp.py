""" Script for computing concept explanations. """
import weakref
import functools
from typing import Dict, List
import torch
import warnings
import zennit
from zennit.core import RemovableHandle, RemovableHandleList

from concept_extraction import _generate_hook


class MaskHook:
    '''Mask hooks for adaptive gradient masking or simple modification.'''

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


# @staticmethod
def mask_map(batch_id: int, concept_ids: List, layer_name=None):
    """
    Wrapper that generates a function thath modifies the gradient (replaced by zennit by attributions).

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


def _append_recording_layer_hooks(model, record_layer):

    handles = []
    layer_out = {}
    record_l_names = record_layer.copy()

    # for l_name in cond_l_names:
    #     if l_name not in record_l_names:
    #         record_l_names.append(l_name)

    # if start_layer is not None and start_layer not in record_l_names:
    #     record_l_names.append(start_layer)

    for name, layer in model.named_modules():

        # if name == self.MODEL_OUTPUT_NAME:
        #     raise ValueError(
        #         "No layer name should match the constant for the identifier of the model output."
        #         "Please change the layer name or the OUTPUT_NAME constant of the object."
        #         "Note, that the condition set then references to the output with OUTPUT_NAME and no longer 'y'.")

        if name in record_l_names:
            h = layer.register_forward_hook(_generate_hook(name, layer_out))
            handles.append(h)
            record_l_names.remove(name)

    # if start_layer in record_l_names:
    #     raise KeyError(f"<start_layer> {start_layer} not found in model.")
    if len(record_l_names) > 0:
        warnings.warn(
            f"Some layer names not found in model: {record_l_names}.")

    return handles, layer_out


def _register_mask_fn(hook, mask_map, b_index, c_indices, l_name):

        if callable(mask_map):
            mask_fn = mask_map(b_index, c_indices, l_name)
        elif isinstance(mask_map, Dict):
            mask_fn = mask_map[l_name](b_index, c_indices, l_name)
        else:
            raise ValueError("<mask_map> must be a dictionary or callable function.")

        hook.fn_list.append(mask_fn)


def compute_concept_explanation(model, data, labels, layer_name, concept_id):
    """ Compute CRP explanations. """

    target = torch.eye(1000, device=data.device)[labels]

    with zennit.attribution.Gradient(model, composite=None) as attributor:

        out, rel = attributor(data, target)


    # Define condition
    # conditions = None   # concept and target
    conditions = [{layer_name: concept_id}]

    hook_map, y_targets, cond_l_names = {}, [], []
    for i, cond in enumerate(conditions):
        for l_name, indices in cond.items():
            # if l_name == MODEL_OUTPUT_NAME:
            #     y_targets.append(indices)
            # else:
            if l_name not in hook_map:
                hook_map[l_name] = MaskHook([])
            _register_mask_fn(hook_map[l_name], mask_map, i, indices, l_name)
            if l_name not in cond_l_names:
                cond_l_names.append(l_name)

    handles, layer_out = _append_recording_layer_hooks(model, cond_l_names)

    name_map = [([name], hook) for name, hook in hook_map.items()]
    # cmask_composite = zennit.composites.NameMapComposite(name_map)

    # if composite is None:
    composite = zennit.composites.EpsilonPlusFlat(canonizers=[zennit.torchvision.VGGCanonizer()])

    mask_composite = zennit.composites.NameLayerMapComposite(
                layer_map=composite.layer_map,
                name_map=name_map,
                canonizers=composite.canonizers,
            )

    with zennit.attribution.Gradient(model, composite=mask_composite) as attributor:

        out, rel = attributor(data, target)

    attribution = rel.cpu().sum(1)
    # with mask_composite.context(model), composite.context(model) as modified:

    #     # if start_layer:
    #     #     _ = modified(data)
    #     #     pred = layer_out[start_layer]
    #     #     grad_mask = self.relevance_init(pred.detach().clone(), None, init_rel)
    #     #     if start_layer in cond_l_names:
    #     #         cond_l_names.remove(start_layer)
    #     #     self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)
    #     #
    #     # else:
    #     pred = modified(data)
    #     grad_mask = self.relevance_init(pred.detach().clone(), y_targets, init_rel)
    #     self.backward(pred, grad_mask, exclude_parallel, cond_l_names, layer_out)

    #     attribution = data.grad.detach()    # self.heatmap_modifier(data, on_device)
        # activations, relevances = {}, {}
        # if len(layer_out) > 0:
        #     activations, relevances = self._collect_hook_activation_relevance(layer_out, on_device)
    [h.remove() for h in handles]

    return attribution  # attrResult(attribution, activations, relevances, pred)

    # for l_name in attr.relevances:
    #     if l_name not in relevances:
    #         relevances[l_name] = attr.relevances[l_name]
    #         activations[l_name] = attr.activations[l_name]
    #     else:
    #         relevances[l_name] = torch.cat([relevances[l_name], attr.relevances[l_name]], dim=0)
    #         activations[l_name] = torch.cat([activations[l_name], attr.activations[l_name]], dim=0)

    # if heatmap is None:
    #     heatmap = attr.heatmap
    #     prediction = attr.prediction
    # else:
    #     heatmap = torch.cat([heatmap, attr.heatmap], dim=0)
    #     prediction = torch.cat([prediction, attr.prediction], dim=0)

    # return attrResult(heatmap, activations, relevances, prediction)