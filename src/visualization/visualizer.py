import os
import sys
from omegaconf import DictConfig
import glob
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("./src")
from encode_dataset import get_dataset
from helpers.concept_visualization import load_concept_attributions, show_concept_examples

class FeatVisHook:

    def __init__(self, FV, concept, layer_name, dict_inputs, on_device):
        """
        Parameters:
            dict_inputs: contains sample_indices and targets inputs to FV.analyze_activation and FV.analyze_relevance 
        """

        self.FV = FV
        self.concept = concept
        self.layer_name = layer_name
        self.dict_inputs = dict_inputs
        self.on_device = on_device

    def backward(self, module, grad):
        '''Hook applied during backward-pass'''

        s_indices, targets = self.dict_inputs["sample_indices"], self.dict_inputs["targets"]
        relevance = grad.detach().to(self.on_device) if self.on_device else grad.detach()
        self.FV.analyze_relevance(relevance, self.layer_name, self.concept, s_indices, targets)

        return grad

    def copy(self):
        '''Return a copy of this hook.
        This is used to describe hooks of different modules by a single hook instance.
        Copies retain the same stored_grads list.
        '''
        return self.__class__(self.FV, self.concept, self.layer_name, self.dict_inputs, self.on_device)

    def remove(self):
        pass

    def register(self, module):
        '''Register this instance by registering the neccessary forward hook to the supplied module.'''
        return RemovableHandleList([
            RemovableHandle(self),
            module.register_forward_hook(self.post_forward),
        ])


class ConceptVis:

    def __init__(self, layer_map, device) -> None:
        self.layer_map = layer_map # conditions
        self.device = device

    def run(self, batches):

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, self.device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        for b in range(batches):

            samples_batch = samples[b * batch_size: (b + 1) * batch_size]
            data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)

            targets_samples = np.array(targets_samples)  # numpy operation needed

            # convert multi target to single target if user defined the method
            data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]
            # dict_inputs is linked to FeatHooks
            dict_inputs["sample_indices"] = sample_indices
            dict_inputs["targets"] = targets

            # composites are already registered before
            self.attribution(data_broadcast, conditions, None, exclude_parallel=False)

            # if b % checkpoint == checkpoint - 1:
            #     self._save_results((last_checkpoint, sample_indices[-1] + 1))
            #     last_checkpoint = sample_indices[-1] + 1

        # TODO: what happens if result arrays are empty?
        self._save_results((last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()


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
        # layers = list(concept_conditions.keys())
        layers = ['features.22']
        # print(layers)
        concepts = concept_conditions['features.21'].cpu().numpy()
        print(f"{pth_file}: {concepts}")

        for concept_id in concepts:
            # attrs, img_ids = load_concept_attributions(concept_file=concept_file_path, layer_name=layers[0], concept_id=concept_id)

            fig = show_concept_examples(dataset, concept_file_path, layers[0], concept_id)
            plt.savefig(os.path.join(results_path, f"{layers[0]}_{concept_id}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
