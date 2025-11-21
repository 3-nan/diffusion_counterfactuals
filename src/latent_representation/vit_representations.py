""" Functions building encoded data representations. """
import torch
from torch.nn.modules.activation import MultiheadAttention
from torchvision.transforms import Compose, CenterCrop, Normalize
import zennit

from .representations import store_hook

# # create a hook to keep track of intermediate outputs
def attention_store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    # print(len(output))
    # print(output[0].size())
    # print(output[1])
    # raise ValueError
    module.output = output[0]
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output[0].retain_grad()

def compute_vit_layer_attributions(classifier, imgs, targets, layers=None):
    """ Compute explanations. """

    # create a composite
    # canonizers = [zennit.torchvision.VGGCanonizer()]
    # composite = zennit.composites.EpsilonGammaBox(0., 1., gamma=0.1, canonizers=canonizers)
    # choose a target class for the attribution
    target = torch.eye(1000)[targets]
    target = target.to(targets.device)

    # print(imgs.size())
    # print(imgs.max())
    transforms = Compose([
        CenterCrop(224),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # transforms = classifier.transforms
    # weights = classifier.weights

    modules = []
    for n, m in classifier.named_modules():
        if n in layers:
            # print(f'Module {n} found')
            modules.append(m)

    # create the attributor, specifying model and composite
    with zennit.attribution.Gradient(model=classifier) as attributor:

        timgs = transforms(imgs)

        handles = []
        for module in modules:
            if isinstance(module, MultiheadAttention):
                handles.append(module.register_forward_hook(attention_store_hook))
            else:
                handles.append(module.register_forward_hook(store_hook))
        # handles = [
        #     module.register_forward_hook(store_hook) for module in modules
        # ]

        # compute the model output and attribution
        output, attribution = attributor(timgs, target)

    for handle in handles:
        handle.remove()

    activation_dict, norm_activation_dict = {}, {}
    attribution_dict, norm_attribution_dict = {}, {}
    rf_neuron_dict = {}

    for name, module in zip(layers, modules):
        act = module.output
        attr = module.output.grad

        # print(act.size())
        # print(attr.size())

        # test some code
        attr = attr.detach().cpu()
        act = act.detach().cpu()

        # print(f'Act size {act.size()} Attr size {attr.size()}')

        if len(act.size()) >= 3:
            rf_neuron = torch.argmax(attr.flatten(start_dim=2), dim=-1)
        else:
            rf_neuron = 0

        if len(act.size()) == 4:
            channel_act = act.sum(axis=[2,3])
            channel_attr = attr.sum(axis=[2,3])
        elif len(act.size()) == 3:
            # channel_act = act.sum(axis=1)
            # channel_attr = attr.sum(axis=1)
            channel_act = act[:, 0, :]
            channel_attr = attr[:, 0, :]
        else:
            channel_act = act
            channel_attr = attr

        # print(f'Channel attr: {channel_attr.size()}')
        # raise ValueError

        norm_channel_act = channel_act / (torch.abs(channel_act).sum(-1).view(-1, 1) + 1e-10)
        norm_channel_attr = channel_attr / (torch.abs(channel_attr).sum(-1).view(-1, 1) + 1e-10)

        activation_dict[name] = channel_act
        norm_activation_dict[name] = norm_channel_act

        attribution_dict[name] = channel_attr
        norm_attribution_dict[name] = norm_channel_attr

        rf_neuron_dict[name] = rf_neuron
    
    return activation_dict, norm_activation_dict, attribution_dict, norm_attribution_dict, rf_neuron_dict
