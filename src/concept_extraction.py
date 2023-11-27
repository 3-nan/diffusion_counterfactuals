""" Script with functionality for extracting relevant concepts. """
import torch
import zennit


# @staticmethod
def _generate_hook(layer_name, layer_out):
    def get_tensor_hook(module, input, output):
        layer_out[layer_name] = output
        output.retain_grad()

    return get_tensor_hook


@staticmethod
def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()


@torch.enable_grad()
def get_latent_representation(classifier, sample, layer, label):

    inp = sample.detach().clone()
    inp.requires_grad = True

    target = torch.eye(1000, device=inp.device)[[label]]

    # print(f"Target size: {target.size()}")

    layer_out = {}

    handles = [
        layer.register_forward_hook(_generate_hook('layer', layer_out))
    ]

    with zennit.attribution.Gradient(classifier, composite=None) as attributor:

        out, rel = attributor(inp, target)

        grad = layer_out['layer'].grad.detach()

    for handle in handles:
        handle.remove()

    return grad

def extract_differing_concepts(repr, cf_repr):
    """ Extract most important and at the same time 
        most differing concepts between both latent 
        space representations.
    """

    # print(repr.size())
    # print(cf_repr.size())
    repr = torch.sum(repr, dim=(2,3))
    cf_repr = torch.sum(cf_repr, dim=(2,3))

    diff = torch.abs(repr - cf_repr)
    # print(f"diff size: {diff.size()}")

    concepts = torch.argsort(diff, dim=1)

    # print(f"concepts size: {concepts.size()}")

    return concepts[:, -2:]

def compute_concept_conditioning(classifier, sample, layer, cf_class):
    """ Compute latent space representations for actual and counterfactual class.
        Compare representations and extract most important and differing concepts.
    """

    # print(f"cf class {cf_class}")

    pred_logits = classifier(sample)

    pred_classes = torch.argmax(pred_logits, axis=1)
    # prob_best_class = pred_logits.sigmoid().detach()

    # Get layer handle
    layer_handle = None
    for n, m in classifier.named_modules():
        if n == layer:
            layer_handle = m

    assert layer_handle

    # Explain classifier
    repr = get_latent_representation(classifier, sample, layer_handle, pred_classes)

    # Explain classifier for counterfactual class
    cf_repr = get_latent_representation(classifier, sample, layer_handle, cf_class)

    # Extract concepts
    concepts = extract_differing_concepts(repr, cf_repr)

    conditions = []
    for sample_concepts in concepts:
        conditions.append({layer: sample_concepts})
    # conditions = [{layer: concepts}]

    return conditions
