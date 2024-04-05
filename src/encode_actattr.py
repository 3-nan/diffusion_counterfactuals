import os
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm

def compute_act_attr_encoding(activation, attribution):
    """ Use the attribution to put a filtering on the activation. """

    # Compute k quantile with 10 percent of overall attribution
    quantile = np.quantile(attribution, 0.9)

    # Use indices as filter
    mask = np.array((attribution < quantile))

    # Restrict activation
    act_repr = activation.copy()
    act_repr[mask] = act_repr[mask] * 0.1

    # Save
    return act_repr

@hydra.main(version_base=None, config_path="../configs/clustering", config_name="v1")
def main(cfg : DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for base in [True]:

        # Settings
        results_path = os.path.join(cfg.output_dir, "data_representation")
        os.makedirs(results_path, exist_ok=True)

        if base:
            activations_file_path = os.path.join(results_path, 'imagenet_acts_base.h5')
            attributions_file_path = os.path.join(results_path, 'imagenet_rels_base.h5')

            actattr_file_path = os.path.join(results_path, 'imagenet_actattr_base.h5')
        else:
            activations_file_path = os.path.join(results_path, 'imagenet_acts_cf.h5')
            attributions_file_path = os.path.join(results_path, 'imagenet_rels_cf.h5')



        inter_layers = cfg.intermediate_layers

        with h5py.File(activations_file_path, 'r', locking=False) as actfile:
            labels = actfile['label'][:]
        print(labels.shape)
        with h5py.File(actattr_file_path, 'a', locking=False) as actattr_file:
            actattr_file.require_dataset(
                'label',
                shape=labels.shape,
                dtype='uint16',
                maxshape=labels.shape,
                chunks=True,
            )
            actattr_file['label'][:] = labels

        with h5py.File(activations_file_path, 'r', locking=False) as actfile:
            predictions = actfile['prediction'][:]
        print(predictions.shape)
        with h5py.File(actattr_file_path, 'a', locking=False) as actattr_file:
            actattr_file.require_dataset(
                'prediction',
                shape=predictions.shape,
                dtype='uint16',
                maxshape=predictions.shape,
                chunks=True,
            )
            actattr_file['prediction'][:] = predictions

        for layer in inter_layers:

            # read acts
            with h5py.File(activations_file_path, 'r', locking=False) as actfile:
                act_shape = actfile['attribution'][layer].shape

            with h5py.File(actattr_file_path, 'a', locking=False) as actattr_file:
                actattr_file.require_group(layer)
                actattr_file[layer].require_dataset(
                    'attribution',
                    shape=act_shape,    # (0,) + act_shape[1:],
                    dtype='float32',
                    maxshape=act_shape,
                    chunks=True,
                )

            for i in tqdm(range(act_shape[0])):

                with h5py.File(activations_file_path, 'r', locking=False) as actfile:
                    act = actfile['attribution'][layer][i]

                # read attrs
                with h5py.File(attributions_file_path, 'r', locking=False) as attrfile:
                    attr = attrfile['attribution'][layer][i]
                
                actattr = compute_act_attr_encoding(act, attr)

                with h5py.File(actattr_file_path, 'a', locking=False) as actattr_file:
                    dset = actattr_file[layer]['attribution']

                    dset[i, :] = actattr

        # compare and compute actattr

        # acts, norm_acts, attrs, norm_attrs, rf_neurons = compute_layer_attributions(classifier_model, image, in_class_pred, layers=inter_layers)

        # # Save representations in h5py
        # append_attributions_to_attribution_database(
        #     activations_file_path,
        #     acts,
        #     norm_acts,
        #     rf_neurons,
        #     np.array(in_class_pred.cpu()),
        #     np.array(label.cpu()))

        # append_attributions_to_attribution_database(
        #     attributions_file_path,
        #     attrs,
        #     norm_attrs,
        #     rf_neurons,
        #     np.array(in_class_pred.cpu()),
        #     np.array(label.cpu()))

if __name__ == "__main__":
    main()
