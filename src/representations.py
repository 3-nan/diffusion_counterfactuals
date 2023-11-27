""" Functions building encoded data representations. """
import torch
import zennit

# create a hook to keep track of intermediate outputs
def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def compute_layer_attributions(classifier, imgs, targets, layers=None):
    """ Compute explanations. """

    # create a composite
    composite = zennit.composites.EpsilonPlusFlat(canonizers=[zennit.torchvision.VGGCanonizer()])

    # choose a target class for the attribution
    target = torch.eye(1000)[targets]
    target = target.to(targets.device)

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

    attribution_dict = {}
    for name, module in zip(layers, modules):
        attr = module.output.grad
        attribution_dict[name] = attr.detach().cpu().sum(axis=[2,3])
    
    return attribution_dict
