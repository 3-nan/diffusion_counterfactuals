""" Functions building encoded data representations. """
import torch
import zennit

# create a hook to keep track of intermediate outputs
def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def compute_layer_attributions(classifier, imgs, targets, layers=None, num_classes=1000):
    """ Compute explanations. """

    # create a composite
    canonizers = [zennit.torchvision.VGGCanonizer()]
    composite = zennit.composites.EpsilonGammaBox(0., 1., gamma=0.1, canonizers=canonizers)
    # choose a target class for the attribution -- num_classes must match the
    # classifier's output width (1000 for ImageNet, 196 for StanfordCars, etc.),
    # otherwise the one-hot target is the wrong length for the backward pass.
    target = torch.eye(num_classes, device=targets.device)[targets]

    modules = []
    for n, m in classifier.named_modules():
        if n in layers:
            modules.append(m)

    # create the attributor, specifying model and composite
    with zennit.attribution.Gradient(model=classifier, composite=composite) as attributor:

        handles = [
            module.register_forward_hook(store_hook) for module in modules
        ]

        # compute the model output and attribution
        output, attribution = attributor(imgs, target)

    for handle in handles:
        handle.remove()

    activation_dict, norm_activation_dict = {}, {}
    attribution_dict, norm_attribution_dict = {}, {}
    rf_neuron_dict = {}

    for name, module in zip(layers, modules):
        act = module.output
        attr = module.output.grad

        # test some code
        attr = attr.detach().cpu()
        act = act.detach().cpu()

        rf_neuron = torch.argmax(attr.flatten(start_dim=2), dim=-1)

        channel_act = act.sum(axis=[2,3])
        channel_attr = attr.sum(axis=[2,3])

        norm_channel_act = channel_act / (torch.abs(channel_act).sum(-1).view(-1, 1) + 1e-10)
        norm_channel_attr = channel_attr / (torch.abs(channel_attr).sum(-1).view(-1, 1) + 1e-10)

        activation_dict[name] = channel_act
        norm_activation_dict[name] = norm_channel_act

        attribution_dict[name] = channel_attr
        norm_attribution_dict[name] = norm_channel_attr

        rf_neuron_dict[name] = rf_neuron
    
    return activation_dict, norm_activation_dict, attribution_dict, norm_attribution_dict, rf_neuron_dict
