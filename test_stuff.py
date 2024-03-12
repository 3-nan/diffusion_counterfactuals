import hydra
import torch
from torch import nn
import zennit

from src.encode_dataset import get_classifier, get_dataset
from src.crp import MaskHook, _register_mask_fn
from src.helpers.concepts import ChannelConcept


def get_attribution(classifier_model, data, labels):

    # Explain predictions
    canonizers = [zennit.torchvision.VGGCanonizer()]

    composite = zennit.composites.EpsilonGammaBox(0., 1., gamma=0.1, canonizers=canonizers)

    target = torch.eye(1000, device=data.device)[labels]

    with zennit.attribution.Gradient(classifier_model, composite=composite) as attributor:

        pred, rels = attributor(data, target)

    return rels.detach().cpu()

def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def get_intermediate_attribution(classifier_model, data, labels, layer_name):

    canonizers = [zennit.torchvision.VGGCanonizer()]

    composite = zennit.composites.EpsilonGammaBox(0., 1., gamma=0.1, canonizers=canonizers)

    target = torch.eye(1000, device=data.device)[labels]

    layers = [layer_name]
    modules = []
    for n, m in classifier_model.named_modules():
        if n in layers:
            modules.append(m)

    with zennit.attribution.Gradient(classifier_model, composite=composite) as attributor:

        handles = [
            module.register_forward_hook(store_hook) for module in modules
        ]

        pred, rels = attributor(data, target)

    for handle in handles:
        handle.remove()
    
    attr_dict = {}
    for name, module in zip(layers, modules):
        attr = module.output.grad

        attr = attr.detach().cpu()

        attr_dict[name] = attr

    return attr_dict

def get_concept_attribution(classifier_model, data, labels, layer_name, concept_id):

    canonizers = [zennit.torchvision.VGGCanonizer()]

    composite = zennit.composites.EpsilonPlus(canonizers=canonizers)
    # composite = zennit.composites.EpsilonGammaBox(0., 1., gamma=0.1, canonizers=canonizers)

    target = torch.eye(1000, device=data.device)[labels]

    for n, m in classifier_model.named_modules():
        if n == layer_name:
            layer_handle = m

    assert layer_handle

    hook_ref = MaskHook([])
    _register_mask_fn(hook_ref, ChannelConcept.mask, 0, [concept_id], layer_name)
    # layer_handle.register_hook

    name_map = [([layer_name], hook_ref)]

    mask_composite = zennit.composites.NameLayerMapComposite(
                layer_map=composite.layer_map,
                name_map=name_map,
                canonizers=composite.canonizers,
            )

    with zennit.attribution.Gradient(classifier_model, composite=mask_composite) as attributor:

        pred, rels = attributor(data, target)

    return rels.detach().cpu()


@hydra.main(version_base=None, config_path="./configs/ldce", config_name="v1")
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    # Load dataset
    dataset = get_dataset(cfg, last_data_idx=0)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False, num_workers=1)

    data, labels, tgt_classes, unique_data_idx = next(iter(data_loader))

    data = data.to(device)
    labels = labels.to(device)

    rels = get_attribution(classifier_model, data, labels)

    for r, rel in enumerate(rels):
        attr = rel.sum(0)

        expl = zennit.image.imgify(attr, cmap='coldnhot', symmetric=True)
        expl.save(f'/results/counterfactuals/test/{r}_expl.png')

    targ_rels = get_attribution(classifier_model, data, tgt_classes)

    for r, rel in enumerate(targ_rels):
        attr = rel.sum(0)

        expl = zennit.image.imgify(attr, cmap='coldnhot', symmetric=True)
        expl.save(f'/results/counterfactuals/test/{r}_targ_expl.png')

    # Get intermediate representation
    prev_layer_name = 'features.14'
    layer_name = 'features.15'

    attr_dict = get_intermediate_attribution(classifier_model, data, labels, prev_layer_name)
    latent_rels = attr_dict[prev_layer_name]
    latent_rels = latent_rels.sum(axis=[2,3])

    for s, (sample, label, latent_rel) in enumerate(zip(data, labels, latent_rels)):

        # sample = 2. * sample - 1

        print(f'{torch.min(sample)} : {torch.max(sample)}')

        # cid = torch.argmax(latent_rel)
        cid = torch.argsort(latent_rel)[-7]
        concept_rel = get_concept_attribution(classifier_model, sample.unsqueeze(0), label.unsqueeze(0), layer_name, cid)

        attr = concept_rel[0].sum(0)
        expl = zennit.image.imgify(attr, cmap='coldnhot', symmetric=True)
        expl.save(f'/results/counterfactuals/test/{s}_concept_expl.png')
    # concept_ids = torch.argsort(rel_c, descending=True)

    # concept_rels = get_concept_attribution(classifier_model, data, labels, layer_name, 7)

    # for r, rel in enumerate(concept_rels):
    #     attr = rel.sum(0)

    #     expl = zennit.image.imgify(attr, cmap='coldnhot', symmetric=True)
    #     expl.save(f'/results/counterfactuals/test/{r}_concept_expl.png')


if __name__ == '__main__':
    main()
