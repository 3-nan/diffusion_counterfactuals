import os
import hydra
import torch

@hydra.main(version_base=None, config_path="../configs/clustering", config_name="v1")
def main(cfg : DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for base in [True, False]:

        # Settings
        results_path = os.path.join(cfg.output_dir, "data_representation")
        os.makedirs(results_path, exist_ok=True)

        if base:
            activations_file_path = os.path.join(results_path, 'imagenet_acts_base.h5')
            attributions_file_path = os.path.join(results_path, 'imagenet_rels_base.h5')
        else:
            activations_file_path = os.path.join(results_path, 'imagenet_acts_cf.h5')
            attributions_file_path = os.path.join(results_path, 'imagenet_rels_cf.h5')



        inter_layers = cfg.intermediate_layers

        for i in range(1000):
        # read acts

        # read attrs

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
