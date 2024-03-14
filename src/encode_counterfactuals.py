""" Script for computing intermediate representations of the samples in the dataset. 
    The intermediate representations are used for finding near misses / near miss clusters.
    Also, prototypical concept representations can be derived from that.
"""
import os
import sys
import random

import hydra
# from hydra.utils import instantiate
import h5py
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import torch
# import torchvision
# from torchvision import transforms
import torchvision.transforms.functional as tf
from tqdm import tqdm

sys.path.append("./ldce")
from ldce.sampling_helpers import normalize
# from ldce.utils.madry_net import MadryNet
# from ldce.utils.preprocessor import Crop, CropAndNormalizer

from representations import compute_layer_attributions
from helpers.data_model_helpers import get_classifier, get_dataset


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_attributions_to_attribution_database(
        attributions_file_path: str,
        attributions_dict: dict,
        norm_attributions_dict: dict,
        rf_neurons_dict: dict,
        predictions: np.ndarray,
        labels: np.ndarray) -> None:
    """Appends the specified attributions to the attributions database.

    Parameters
    ----------
        attributions_file_path: str
            The path to the attributions database file, to which the attributions are to be appended. If the
            attributions database file does not exist, yet, it is created.
        attributions: numpy.ndarray
            The attributions that are to be appended to the attributions database.
        predictions: numpy.ndarray
            The predictions of the classifier for which the attributions were computed. that are to be appended to the
            attributions database.
        labels: numpy.ndarray
            The ground-truth labels of the samples from which the attributions were computed, that are to be appended to
            the attributions database.
    """

    # If the attributions file does not exist, yet, it is created (the HDF5 datasets for the attributions, their
    # predictions, and their ground-truth labels are created, but they are made resizable, so that each time
    # attributions are appended, the HDF5 datasets can be resized to accommodate the new attributions)
    if not os.path.exists(attributions_file_path):
        # attribution_shape = tuple(attributions.shape[1:])
        number_of_predictions = tuple(predictions.shape[1:])
        with h5py.File(attributions_file_path, 'w', locking=False) as attributions_file:
            attr_group = attributions_file.create_group('attribution')
            norm_attr_group = attributions_file.create_group('norm_attribution')
            neuron_group = attributions_file.create_group('rf_neuron')

            for layer_name, attrs in attributions_dict.items():
                attribution_shape = tuple(attrs.shape[1:])
                attr_group.create_dataset(
                    layer_name,
                    shape=(0,) + attribution_shape,
                    dtype='float32',
                    maxshape=(None,) + attribution_shape,
                    chunks=True,
                    compression='gzip',
                )
                norm_attr_group.create_dataset(
                    layer_name,
                    shape=(0,) + attribution_shape,
                    dtype='float32',
                    maxshape=(None,) + attribution_shape,
                    chunks=True,
                    compression='gzip',
                )

            for layer_name, rf_neurons in rf_neurons_dict.items():
                neuron_shape = tuple(rf_neurons.shape[1:])
                neuron_group.create_dataset(
                    layer_name,
                    shape=(0,) + neuron_shape,
                    dtype='uint16',
                    maxshape=(None,) + neuron_shape,
                    chunks=True,
                    compression='gzip',
                )

            attributions_file.create_dataset(
                'prediction',
                shape=(0,) + number_of_predictions,
                dtype='uint16',
                maxshape=(None,) + number_of_predictions,
                chunks=True,
                compression='gzip'
            )
            attributions_file.create_dataset(
                'label',
                shape=(0,),
                dtype='uint16',
                maxshape=(None,),
                chunks=True,
                compression='gzip'
            )

    # Appends the attributions, their predictions, and their ground-truth labels to the attributions file
    with h5py.File(attributions_file_path, 'a', locking=False) as attributions_file:

        # Determines how many attributions are currently in the dataset and how many attributions are being added (this
        # is needed to correctly resize the HDF5 datasets)
        number_of_existing_attributions = attributions_file['prediction'].shape[0]  # pylint: disable=no-member
        number_of_new_attributions = predictions.shape[0]

        # Resizes the HDF5 datasets for the attributions, their predictions, and their ground-truth labels, so that
        # they can fit the attributions that are to be appended
        for layer_name in attributions_dict:
            attributions_file['attribution'][layer_name].resize(  # pylint: disable=no-member
                number_of_existing_attributions + number_of_new_attributions,
                axis=0
            )
            attributions_file['norm_attribution'][layer_name].resize(  # pylint: disable=no-member
                number_of_existing_attributions + number_of_new_attributions,
                axis=0
            )
            attributions_file['rf_neuron'][layer_name].resize(  # pylint: disable=no-member
                number_of_existing_attributions + number_of_new_attributions,
                axis=0
            )
        attributions_file['prediction'].resize(  # pylint: disable=no-member
            number_of_existing_attributions + number_of_new_attributions,
            axis=0
        )
        attributions_file['label'].resize(  # pylint: disable=no-member
            number_of_existing_attributions + number_of_new_attributions,
            axis=0
        )

        # Appends the attributions, their predictions, and their ground-truth labels to the attributions file
        for layer_name, attrs in attributions_dict.items():
            attributions_file['attribution'][layer_name][number_of_existing_attributions:] = attrs
        for layer_name, attrs in norm_attributions_dict.items():
            attributions_file['norm_attribution'][layer_name][number_of_existing_attributions:] = attrs
        for layer_name, rf_neurons in rf_neurons_dict.items():
            attributions_file['rf_neuron'][layer_name][number_of_existing_attributions:] = rf_neurons
        attributions_file['prediction'][number_of_existing_attributions:] = predictions
        attributions_file['label'][number_of_existing_attributions:] = labels


@hydra.main(version_base=None, config_path="../configs/clustering", config_name="v1")
def main(cfg : DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for base in [True, False]:

    counterfactuals_dir = cfg.counterfactuals_dir

    # Settings
    results_path = os.path.join(cfg.output_dir, "data_representation")
    os.makedirs(results_path, exist_ok=True)

    # if base:
    #     activations_file_path = os.path.join(results_path, 'imagenet_acts_base.h5')
    #     attributions_file_path = os.path.join(results_path, 'imagenet_rels_base.h5')
    # else:
    activations_file_path = os.path.join(results_path, 'imagenet_acts_cf_gen.h5')
    attributions_file_path = os.path.join(results_path, 'imagenet_rels_cf_gen.h5')

    if os.path.exists(activations_file_path):
        os.remove(activations_file_path)
    if os.path.exists(attributions_file_path):
        os.remove(attributions_file_path)

    # Load model
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    # classifier_model.train = disabled_train
    print([f"{n}: {type(m)}" for n,m in classifier_model.named_modules()])

    # Load dataset
    # n_samples_per_class = cfg.n_samples_per_class
    batch_size = 32 # cfg.data.batch_size
    # shuffle = cfg.get("shuffle", False)

    print(f'{cfg.data.start_sample} -> {cfg.data.end_sample}')
    last_data_idx = 0
    dataset = get_dataset(cfg, last_data_idx=last_data_idx, base=False)
    print(type(dataset))
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Iterate dataset (max 100 samples per class?)

    for i, batch in enumerate(tqdm(data_loader)):

        if "fixed_seed" in cfg:
            set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
            seed = seed if cfg.fixed_seed else -1
            
        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            tgt_classes = tgt_classes.to(device) #squeeze()
        else:
            image, label, unique_data_idx = batch
            if "ImageNet" in cfg.data._target_:
                tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
            # elif "CelebAHQDataset" in cfg.data._target_:
            #     tgt_classes = (1 - label).type(torch.float32)
            # elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
            #     tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError

        # Load generated counterfactual images
        file_paths = [os.path.join(counterfactuals_dir, f'{str(udx).zfill(5)}.png') for udx in unique_data_idx.numpy()]


        cf_imgs = [dataset.transform(pil_loader(fp)) for fp in file_paths]
        cf_imgs = torch.stack(cf_imgs, dim=0)

        # encode counterfactuals
        image = cf_imgs
        label = tgt_classes

        # raise ValueError
        image = image.to(device)
        label = label.to(device)

        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(image)
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                # x = _map_img(x)
                x = image
                if "classifier_wrapper" not in cfg.classifier_model: # only works for ImageNet!
                    x = tf.center_crop(x, 224)
                    x = normalize(x)
                logits = classifier_model(x)
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                # in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                # in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            # print("in class_pred: ", in_class_pred, in_confid)
        
        # for j, l in enumerate(label):
        #     print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")

        # Optional: Compute intermediate activation

        # Compute intermediate attributions
        inter_layers = cfg.intermediate_layers

        acts, norm_acts, attrs, norm_attrs, rf_neurons = compute_layer_attributions(classifier_model, image, in_class_pred, layers=inter_layers)

        # Save representations in h5py
        append_attributions_to_attribution_database(
            activations_file_path,
            acts,
            norm_acts,
            rf_neurons,
            np.array(in_class_pred.cpu()),
            np.array(label.cpu()))

        append_attributions_to_attribution_database(
            attributions_file_path,
            attrs,
            norm_attrs,
            rf_neurons,
            np.array(in_class_pred.cpu()),
            np.array(label.cpu()))

if __name__ == "__main__":
    main()
