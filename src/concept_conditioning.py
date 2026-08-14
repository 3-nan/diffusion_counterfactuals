""" Functionality for deriving conditioning to single concepts in the gradient backward pass. """
import sys
import numpy as np
import torch
import torchvision.transforms.functional as tf
import zennit
sys.path.append("./")
from src.sampling_helpers import normalize


def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def compute_concept_conditioning(model, image, target, layer_name, num_concepts=20, num_classes=1000, spatial=False, spatial_th=0.4, cond_option='sumabs', return_gradient=False, classifier_wrapper=False):
    """ Compute concept conditioning. """

    # print(f'image min: {torch.min(image)} and max {torch.max(image)}')
    # image = _map_img(image)
    if not classifier_wrapper: # only works for ImageNet-style 224-native classifiers!
        image = tf.center_crop(image, 224)
        image = normalize(image)
    # else: model (e.g. Normalizer(...)) already crops/normalizes internally --
    # doing it again here would both reintroduce a hardcoded 224 crop for
    # non-224-native classifiers and double-apply ImageNet normalization.

    layer = None
    for name, lay in model.named_modules():
        # print(f'{name} : {type(lay)}')
        # exact match for unwrapped classifiers; suffix match so layer_name
        # (e.g. "features.37") still resolves when model is a wrapper like
        # Normalizer(classifier) that nests it under "classifier."
        if name == layer_name or name.endswith("." + layer_name):
            layer = lay

    assert layer

    single_label = False

    if single_label:
        target = target.type(torch.int64)
        target_tensor = torch.eye(num_classes, device=image.device)[target, 0]
    else:
        target_tensor = torch.eye(num_classes, device=image.device)[target]
    # target_tensor = target_tensor.to(image.device)

    # Get gradient in specified layer
    with zennit.attribution.Gradient(model=model) as attributor:

        handles = []
        # for n,l in attributor.named_modules():
        #     if n == "layer_name":
        handles.append(layer.register_forward_hook(store_hook))

        # compute the model output and gradient
        output, attr = attributor(image, target_tensor)

    for handle in handles:
        handle.remove()

    # print the gradient tensors for demonstration
    grad = layer.output.grad.detach()
    # print(f'Layer output grad size: {grad.size()} in layer {layer_name} {type(layer)}')

    # print(type(model))

    # Extract most important channels based on the selected cond_option
    if cond_option == "sumabs":
        if len(grad.size()) == 4:
            channel_grads = grad.cpu().sum((2,3)).numpy()
        else:
            channel_grads = grad.cpu()[:,0,:].numpy()
        conditions = [np.argsort(np.abs(cg))[-num_concepts:] for cg in channel_grads]
        conditions = np.array(conditions)

        diff = []
        for conds, cgrads in zip(conditions, channel_grads):
            diff.append(cgrads[conds])
        diff = np.array(diff)

    elif cond_option == "sum":
        channel_grads = grad.cpu().sum((2,3)).numpy()
        conditions = [np.argsort(cg)[-num_concepts:] for cg in channel_grads]
        conditions = np.array(conditions)

        diff = []
        for conds, cgrads in zip(conditions, channel_grads):
            diff.append(cgrads[conds])
        diff = np.array(diff)

    elif cond_option == "sumequal":
        nce = int(num_concepts / 2)
        channel_grads = grad.cpu().sum((2,3)).numpy()
        conditions = [np.concatenate((np.argsort(cg)[:nce], np.argsort(cg)[-nce:])) for cg in channel_grads]
        conditions = np.array(conditions)

        diff = []
        for conds, cgrads in zip(conditions, channel_grads):
            diff.append(cgrads[conds])
        diff = np.array(diff)
    
    elif cond_option == "absmean":
        channel_grads = grad.cpu().abs().mean((2,3)).numpy()
        conditions = [np.argsort(cg)[-num_concepts:] for cg in channel_grads]
        conditions = np.array(conditions)
        # diff = grad.cpu().mean((2,3)).numpy()[conditions]
        diff = []
        for conds, cgrads in zip(conditions, grad.cpu().mean((2,3)).numpy()):
            diff.append(cgrads[conds])
        diff = np.array(diff)

    elif cond_option == "abssum":
        channel_grads = grad.cpu().abs().sum((2,3)).numpy()
        conditions = [np.argsort(cg)[-num_concepts:] for cg in channel_grads]
        conditions = np.array(conditions)
        # diff = grad.cpu().sum((2,3)).numpy()[conditions]
        diff = []
        for conds, cgrads in zip(conditions, grad.cpu().sum((2,3)).numpy()):
            diff.append(cgrads[conds])
        diff = np.array(diff)

    elif cond_option == "absmax":
        channel_grads = grad.cpu().abs().amax(dim=(2,3)).numpy()
        conditions = [np.argsort(cg)[-num_concepts:] for cg in channel_grads]
        conditions = np.array(conditions)
        diff = []
        for conds, cgrads in zip(conditions, channel_grads):
            diff.append(cgrads[conds])
        diff = np.array(diff)


    else:
        raise NotImplementedError

    if spatial:
        # Spatial filtering
        spatial_cond_mask = torch.zeros_like(grad)
        for s in range(grad.size()[0]):

            for cond in conditions[s]:

                th = grad[s, cond, :, :].abs().max() * spatial_th

                spatial_cond_mask[s, cond, :, :] = (grad[s, cond, :, :].abs() >= th)
        
        # print(f'Spatial cond mask size: {spatial_cond_mask.size()}')
        if return_gradient:
            return {layer_name: spatial_cond_mask}, {layer_name: conditions}, diff, grad.cpu().numpy()
        else:
            return {layer_name: spatial_cond_mask}, {layer_name: conditions}, diff

    # print(conditions)
    # print(f'Condition shape: {conditions.shape}')
    if return_gradient:
        return {layer_name: spatial_cond_mask}, {layer_name: conditions}, diff, grad.cpu().numpy()
    else:
        return {layer_name: conditions}, {layer_name: conditions}, diff
