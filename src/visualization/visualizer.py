import os
import sys
from omegaconf import DictConfig
import glob
import hydra
import matplotlib.pyplot as plt
import torch

sys.path.append("./src")
from encode_dataset import get_dataset
from helpers.concept_visualization import load_concept_attributions, show_concept_examples


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig):

    results_path = os.path.join(cfg.output_dir, "data_representation")
    assert os.path.isdir(results_path)
    attributions_file_path = os.path.join(results_path, 'imagenet_rels.h5')

    concept_file_path = os.path.join(results_path, 'concepts.h5')

    # concept_file = None
    layer_name = None
    concept_id = None

    dataset = get_dataset(cfg)

    out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")

    # collect pth files
    pth_files = glob.glob(out_dir + "/*[0-9].pth")

    # read pth files
    for pth_file in pth_files:
        datadict = torch.load(pth_file)

        concept_conditions = datadict['concept_conditions'][0]
        # print(concept_conditions)
        layers = list(concept_conditions.keys())
        # print(layers)
        concepts = concept_conditions[layers[0]].cpu().numpy()
        print(f"{pth_file}: {concepts}")

        for concept_id in concepts:
            # attrs, img_ids = load_concept_attributions(concept_file=concept_file_path, layer_name=layers[0], concept_id=concept_id)

            fig = show_concept_examples(dataset, concept_file_path, layers[0], concept_id)
            plt.savefig(os.path.join(results_path, f"{layers[0]}_{concept_id}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
