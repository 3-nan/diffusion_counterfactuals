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


def append_concepts_to_concept_database(concept_file_path, layer_name, concept_id, attrs, img_inds):
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


def load_layer_attribution(attributions_file_path, layer_name):
    """ Load the attributions for the specified layer. """

    with h5py.File(attributions_file_path, 'r', locking=False) as attributions_file:
        layer_attr = attributions_file['attribution'][layer_name][:,:]

    return layer_attr

@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_k = 10

    results_path = os.path.join(cfg.output_dir, "data_representation")
    assert os.path.isdir(results_path)
    attributions_file_path = os.path.join(results_path, 'imagenet_rels.h5')

    concept_file_path = os.path.join(results_path, 'concepts.h5')

    with h5py.File(attributions_file_path, 'r', locking=False) as attributions_file:
        attr_group = attributions_file['attribution']
        layer_names = list(attr_group.keys())

        predictions = attributions_file['prediction'][:]
        print(f"Maximizing concepts in layers: {layer_names}")

    dataset = get_dataset(cfg)
    classifier = get_classifier(cfg, device)
    classifier.to(device).eval()

    for layer_name in layer_names:

        print(layer_name)
        layer_attr = load_layer_attribution(attributions_file_path, layer_name)

        # print(f"{layer_name}: {layer_attr.shape}")
        # print([f"{n}: {type(m)}" for n, m  in classifier.named_modules()])

        # For each concept -> extract highest k samples
        for cid in tqdm(range(layer_attr.shape[1])):

            # print(np.argmax(layer_attr[:, i]))

            top_inds = np.argpartition(layer_attr[:, cid], -top_k)[-top_k:]

            # Retrieve samples
            top_samples = [dataset[ti][0] for ti in top_inds]
            preds = predictions[top_inds]

            data = torch.stack(top_samples).to(device)

            # Run CRP for selected samples: concept i, top_inds
            cexpl = compute_concept_explanation(classifier, data, preds, layer_name, cid)

            # Save results
            append_concepts_to_concept_database(
                concept_file_path,
                layer_name,
                cid,
                np.array(cexpl),
                top_inds,
            )
                # layer_attr,
                # i,
                # cexpl,
                # cvis_rf,
                # np.array(in_class_pred.cpu()),
                # np.array(label.cpu()))

if __name__ == '__main__':
    main()
