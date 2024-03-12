""" Module for providing different generation targets. """
import os
import h5py
import hydra
import json
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.distance import euclidean
import torch
import yaml
import zennit

from concept_maximization import maximize_concepts_for_class
from encode_dataset import get_dataset, get_classifier
from helpers.concept_visualization import show_concept_examples
from representations import store_hook


TARGET_MODES = [
    'target_class',
    'ref_sample',
    'ref_latent'
]

def target_selection(encodings_file, layer_name, mode='target_class', feat_method="relevance"):
    """ Main target selection function. """
    
    assert mode in TARGET_MODES

    # Options:
    # A: class based
    # B: reference sample based
    #   1: single ref sample
    #   2: multiple samples
    # C: latent representation
    #
    # A: gradient
    # B: relevance
    # C: activation

    # Params:
    # num_samples to consider?
    # num_concepts
    # concept_ids?

    if mode=='target_class':
        print(f"Mode {mode} selected.")

        # default from file (wordnet) -> tgt_classes

        # class of near miss/cluster (?)

    elif mode == 'ref_sample':
        print(f"Mode {mode} selected.")

        # latent_sample = encode_sample(model, sample, layer_name, label)
        latent_sample = retrieve_sample_encoding()

        # Near Miss
        near_misses = get_near_miss(data_base, latent_sample, layer_name, k=1)

        # Near Cluster
        # near_cluster = get_near_cluster(latent_sample, k=1)

        # Selected samples

def get_nearest(attr, base_attrs, k=1, dist_fn=euclidean):

    # print(base_attrs[0].shape)

    distances = [dist_fn(attr, base_attr) for base_attr in base_attrs]

    # print(distances[:7])
    # print(len(distances))

    return np.argmin(distances)

def compute_gradient_to_class(model, sample, target, rec_layers=None):

    if rec_layers:
        modules = []
        for n, m in model.named_modules():
            if n in rec_layers:
                modules.append(m)

    target_tensor = torch.eye((1000))[target]
    target_tensor = target_tensor.unsqueeze(0).to(sample.device)

    with zennit.attribution.Gradient(model) as attributor:
        handles = [
            module.register_forward_hook(store_hook) for module in modules
        ]
        out, grad = attributor(sample, target_tensor)

    for handle in handles:
        handle.remove()

    grad_dict = {}

    for name, module in zip(rec_layers, modules):
        attr = module.output.grad

        # test some code
        attr = attr.detach().cpu()

        grad_dict[name] = attr

    return grad_dict

@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths: Activations & Attributions
    acts_sample_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_acts_samples.h5')
    attr_sample_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_rels_samples.h5')

    acts_base_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_acts_base.h5')
    attr_base_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_rels_base.h5')

    concept_file_path = os.path.join(cfg.output_dir, 'data_representation', 'base_concepts.h5')

    os.makedirs(os.path.join(cfg.output_dir, 'concept_selection'), exist_ok=True)
    conditions_file = os.path.join(cfg.output_dir, 'concept_selection', 'conditions.yml')

    # Load dataset
    dataset = get_dataset(cfg, base=True)
    classifier = get_classifier(cfg, device)
    classifier.to(device).eval()

    sample_dataset = get_dataset(cfg, base=False)

    # Additional parameters
    layer_name = 'features.27'
    num_concepts = 10

    # Read data from files
    with h5py.File(acts_sample_file_path, 'r', locking=False) as acts_sample_file:

        sample_acts = acts_sample_file['norm_attribution'][layer_name][:,:]
        sample_classes = acts_sample_file['prediction'][:]

    with h5py.File(acts_base_file_path, 'r', locking=False) as acts_base_file:

        base_acts = acts_base_file['norm_attribution'][layer_name][:,:]
        base_classes = acts_base_file['prediction'][:]

    with h5py.File(attr_sample_file_path, 'r', locking=False) as attrs_sample_file:

        sample_attrs = attrs_sample_file['norm_attribution'][layer_name][:,:]

    with h5py.File(attr_base_file_path, 'r', locking=False) as attrs_base_file:

        base_attrs = attrs_base_file['norm_attribution'][layer_name][:,:]

    conditions = {}
    # Iterate samples
    for i, sample_attr in enumerate(sample_acts):

        c_class_idcs = np.where(base_classes != sample_classes[i])[0]

        # Find nearest
        near_idx = get_nearest(sample_attr, base_acts[c_class_idcs], k=1)

        near_attr_idx = get_nearest(sample_attrs[i], base_attrs[c_class_idcs], k=1)

        s_idx = c_class_idcs[near_idx]
        s_attr_idx = c_class_idcs[near_attr_idx]

        print(f"Sample class: {sample_classes[i]}")
        print(f"Target class: {base_classes[s_idx]}")
        print(f"Target attr class: {base_classes[s_attr_idx]}")

        base_act = base_acts[s_idx]


        sample_concepts = np.argsort(sample_attrs[i])[-num_concepts:][::-1]
        target_concepts = np.argsort(base_attrs[s_attr_idx])[-num_concepts:][::-1]


        diff_concepts = sample_attrs[i] - base_attrs[s_attr_idx]
        sorted_diff_concepts = np.argsort(diff_concepts)

        view_concepts = np.concatenate((sorted_diff_concepts[:7], sorted_diff_concepts[-7:]), axis=0)
        print(view_concepts)

        maximize_concepts_for_class(attr_base_file_path, concept_file_path, dataset, classifier, layer_name, base_classes[s_attr_idx], sorted_diff_concepts[:7])
        maximize_concepts_for_class(attr_base_file_path, concept_file_path, dataset, classifier, layer_name, sample_classes[i], sorted_diff_concepts[-7:])

        # diff_concepts = np.argsort(np.abs(sample_attrs[i] - base_attrs[s_attr_idx]))[-num_concepts:][::-1]

        print(f"Sample concepts: {sample_concepts}")
        print(f"Target concepts: {target_concepts}")
        # print(f"Diff concepts: {diff_concepts}")
        print(f"View concepts: {view_concepts}")

        # Compute gradient w.r.t target class in concept layer
        sample = sample_dataset[i][0]
        sample = sample.unsqueeze(0).to(device)
        grad_dict = compute_gradient_to_class(classifier, sample, int(base_classes[s_attr_idx]), rec_layers=[layer_name])
        target_gradient = grad_dict[layer_name]

        grad_concepts = target_gradient.sum((2,3))[0].numpy()
        mean_grad_concepts = target_gradient.abs().mean((2,3))[0].numpy()
        grad_concepts = np.argsort(np.abs(grad_concepts))[-10:]
        mean_grad_concepts = np.argsort(mean_grad_concepts)[-10:]

        print(f"Gradient concepts: {grad_concepts[::-1]}")
        print(f"Mean Gradient concepts: {mean_grad_concepts[::-1]}")

        # raise ValueError
        
        # Show or compute AttrMax for diff_concepts?
        # Show or compute local CRP for diff_concepts?

        # for c, concept_id in enumerate(view_concepts):
        verbose=False
        if verbose:
            for c, concept_id in enumerate(grad_concepts):

                if c < 7:
                    label = int(base_classes[s_attr_idx])
                else:
                    label = int(sample_classes[i])

                if not os.path.isfile(os.path.join(cfg.output_dir, 'concept_selection', f"{layer_name}_{concept_id}_{label}.png")):
                    fig = show_concept_examples(dataset, concept_file_path, layer_name, label, concept_id)
                    plt.savefig(os.path.join(cfg.output_dir, 'concept_selection', f"{layer_name}_{concept_id}_{label}.png"))
                    plt.close(fig)

        # Compare activations

        # Compare attributions

        # Write conditions to file
        conditions.update({
            i: {
                # layer_name: diff_concepts.tolist(),
                # layer_name: view_concepts.tolist(),
                layer_name: grad_concepts.tolist(),
                'y': int(base_classes[s_attr_idx]),
            }
        })

        with open(conditions_file, 'a') as outfile:
            yaml.dump(conditions, outfile, default_flow_style=False)

        with open(os.path.join(cfg.output_dir, 'concept_selection', f'conditions_{layer_name}.json'), 'w') as json_file:
            json_file.write(json.dumps(conditions))


        if i > 21:
            raise ValueError


if __name__ == '__main__':
    main()
