""" Functions for running concept maximization. """
import os
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from encode_dataset import get_dataset, get_classifier
from crp import compute_concept_explanation


def maximize_concepts_for_class(attr_file_path, concept_file_path, dataset, classifier, layer, label, concept_ids, k=10):
    """ For a given label, layer, and concept_id, find and explain samples with maximum concept attribution. """

    label = int(label)
    concept_ids = concept_ids.tolist()
    print(f"maximize {label} for concepts {concept_ids}")

    if os.path.isfile(concept_file_path):
        with h5py.File(concept_file_path, 'r', locking=False) as cf:
            if layer in cf:
                if str(label) in cf[layer]:
                    for concept_id in concept_ids:
                        if str(concept_id) in cf[layer][str(label)]:
                            concept_ids.remove(concept_id)

    with h5py.File(attr_file_path, 'r', locking=False) as abf:

        class_ids = np.where(abf['prediction'][:].astype(int) == label)[0]

        for concept_id in concept_ids:

            cmax_ids = np.argsort(abf['norm_attribution'][layer][class_ids, concept_id])[::-1]
            cmax_ids = cmax_ids[:k]

            cmax_ids = class_ids[cmax_ids]

            top_samples = [dataset[ti][0] for ti in cmax_ids]
            # preds = predictions[top_inds]
            preds = np.repeat(label, len(cmax_ids))

            data = torch.stack(top_samples).to(next(classifier.parameters()).device)

            # Run CRP for selected samples: concept i, top_inds
            cexpl = compute_concept_explanation(classifier, data, preds, layer, concept_id, rf_neurons=None)

            append_concepts_to_concept_database(
                concept_file_path,
                layer,
                preds,
                concept_id,
                np.array(cexpl),
                cmax_ids,
            )

def append_concepts_to_concept_database(concept_file_path, layer, preds, concept_id, attrs, img_inds):

    with h5py.File(concept_file_path, 'a', locking=False) as cf:

        attribution_shape = tuple(attrs.shape)

        layer_group = cf.require_group(layer)
        label_group = layer_group.require_group(str(preds[0]))
        concept_group = label_group.require_group(str(concept_id))

        concept_group.require_dataset(
            'attribution',
            shape=attribution_shape,
            dtype='float32',
            chunks=True,
            compression='gzip',
        )
        concept_group.require_dataset(
            'img_id',
            shape=(attribution_shape[0],),
            dtype='uint32',
            chunks=True,
            compression='gzip',
        )

        concept_group['attribution'][:] = attrs
        concept_group['img_id'][:] = img_inds


def old_append_concepts_to_concept_database(concept_file_path, layer_name, concept_id, attrs, img_inds):
    """ Save attributions to database. """

    if not os.path.exists(concept_file_path):
        attribution_shape = tuple(attrs.shape[1:])
        # number_of_predictions = tuple(attrs.shape[0])
        with h5py.File(concept_file_path, 'w', locking=False) as concept_file:

            layer_group = concept_file.create_group(layer_name)
            concept_group = layer_group.create_group(str(concept_id))

            concept_group.create_dataset(
                'attribution',
                shape=(0,) + attribution_shape,
                dtype='float32',
                maxshape=(None,) + attribution_shape,
                chunks=True,
                compression='gzip',
            )

            concept_group.create_dataset(
                'img_id',
                shape=(0,),
                dtype='uint16',
                maxshape=(None,),
                chunks=True,
                compression='gzip',
            )

    # Appends the attributions, their predictions, and their ground-truth labels to the attributions file
    with h5py.File(concept_file_path, 'a', locking=False) as concept_file:

        layer_group = concept_file.require_group(layer_name)
        concept_group = layer_group.require_group(str(concept_id))
        # Determines how many attributions are currently in the dataset and how many attributions are being added (this
        # is needed to correctly resize the HDF5 datasets)

        attribution_shape = tuple(attrs.shape[1:])
        number_of_new_attributions = attrs.shape[0]

        concept_group.require_dataset(
                'attribution',
                shape=(0,) + attribution_shape,
                dtype='float32',
                maxshape=(None,) + attribution_shape,
                chunks=True,
                compression='gzip',
            )
        concept_group.require_dataset(
                'img_id',
                shape=(0,),
                dtype='uint16',
                maxshape=(None,),
                chunks=True,
                compression='gzip',
            )
        
        concept_group['attribution'].resize(
            number_of_new_attributions, axis=0
        )
        concept_group['img_id'].resize(
            number_of_new_attributions, axis=0
        )

        concept_group['attribution'][:] = attrs
        concept_group['img_id'][:] = img_inds


def load_layer_attribution(attributions_file_path, layer_name, key='norm_attribution'):
    """ Load the attributions for the specified layer. """

    with h5py.File(attributions_file_path, 'r', locking=False) as attributions_file:

        assert attributions_file[key]

        layer_attr = attributions_file[key][layer_name][:,:]
        rf_neurons = attributions_file['rf_neuron'][layer_name][:,:]

    return layer_attr, rf_neurons

@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_k = 12

    results_path = os.path.join(cfg.output_dir, "data_representation")
    assert os.path.isdir(results_path)
    attributions_file_path = os.path.join(results_path, 'imagenet_rels_base.h5')

    concept_file_path = os.path.join(results_path, 'base_concepts.h5')

    with h5py.File(attributions_file_path, 'r', locking=False) as attributions_file:
        attr_group = attributions_file['attribution']
        layer_names = list(attr_group.keys())

        predictions = attributions_file['prediction'][:]
        print(f"Maximizing concepts in layers: {layer_names}")

    dataset = get_dataset(cfg, base=True)
    classifier = get_classifier(cfg, device)
    classifier.to(device).eval()

    # layer_names = ['features.22']

    for layer_name in layer_names:

        print(layer_name)
        layer_attr, rf_neurons = load_layer_attribution(attributions_file_path, layer_name, key='norm_attribution')

        # print(f"{layer_name}: {layer_attr.shape}")
        # print([f"{n}: {type(m)}" for n, m  in classifier.named_modules()])

        # For each concept -> extract highest k samples
        for cid in tqdm(range(layer_attr.shape[1])):

            # print(np.argmax(layer_attr[:, i]))

            # top_inds = np.argpartition(layer_attr[:, cid], -top_k)[-top_k:]
            top_inds = np.argsort(layer_attr[:, cid])[-top_k:]

            concept_neurons = rf_neurons[:, cid][top_inds]

            # Retrieve samples
            top_samples = [dataset[ti][0] for ti in top_inds]
            preds = predictions[top_inds]

            data = torch.stack(top_samples).to(device)

            # Run CRP for selected samples: concept i, top_inds
            cexpl = compute_concept_explanation(classifier, data, preds, layer_name, cid, rf_neurons=None)
            # cexpl = compute_concept_explanation(classifier, data, preds, layer_name, cid, rf_neurons=concept_neurons)

            # Save results
            append_concepts_to_concept_database(
                concept_file_path,
                layer_name,
                cid,
                np.array(cexpl),
                top_inds,
            )


if __name__ == '__main__':
    main()
