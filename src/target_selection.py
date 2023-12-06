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

from encode_dataset import get_dataset
from helpers.concept_visualization import show_concept_examples


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

@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:

    # File paths: Activations & Attributions
    acts_sample_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_acts_samples.h5')
    attr_sample_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_rels_samples.h5')

    acts_base_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_acts_base.h5')
    attr_base_file_path = os.path.join(cfg.output_dir, 'data_representation', 'imagenet_rels_base.h5')

    layer_name = 'features.22'
    concept_file_path = os.path.join(cfg.output_dir, 'data_representation', 'concepts.h5')

    os.makedirs(os.path.join(cfg.output_dir, 'concept_selection'), exist_ok=True)
    conditions_file = os.path.join(cfg.output_dir, 'concept_selection', 'conditions.yml')

    dataset = get_dataset(cfg, base=True)

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

        # print(near_idx)
        # print(c_class_idcs[near_idx])

        s_idx = c_class_idcs[near_idx]
        s_attr_idx = c_class_idcs[near_attr_idx]

        print(f"Sample class: {sample_classes[i]}")
        print(f"Target class: {base_classes[s_idx]}")
        print(f"Target attr class: {base_classes[s_attr_idx]}")

        base_act = base_acts[s_idx]


        sample_concepts = np.argsort(sample_attrs[i])[-10:][::-1]

        target_concepts = np.argsort(base_attrs[s_attr_idx])[-10:][::-1]

        diff_concepts = np.argsort(np.abs(sample_attrs[i] - base_attrs[s_attr_idx]))[-10:][::-1]

        print(f"Sample concepts: {sample_concepts}")
        print(f"Target concepts: {target_concepts}")
        print(f"Diff concepts: {diff_concepts}")
        
        # Show or compute AttrMax for diff_concepts?
        # Show or compute local CRP for diff_concepts?

        # for concept_id in diff_concepts:
        #     fig = show_concept_examples(dataset, concept_file_path, layer_name, concept_id)
        #     plt.savefig(os.path.join(cfg.output_dir, 'concept_selection', f"{layer_name}_{concept_id}.png"))
        #     plt.close(fig)

        # Compare activations

        # Compare attributions

        # Write conditions to file
        conditions.update({
            i: {
                layer_name: diff_concepts.tolist(),
                'y': int(base_classes[s_attr_idx]),
            }
        })

        with open(conditions_file, 'a') as outfile:
            yaml.dump(conditions, outfile, default_flow_style=False)

        with open(os.path.join(cfg.output_dir, 'concept_selection', 'conditions.json'), 'w') as json_file:
            json_file.write(json.dumps(conditions))


        if i > 15:
            raise ValueError


if __name__ == '__main__':
    main()
