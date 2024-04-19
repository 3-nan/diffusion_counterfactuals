""" Evaluation study for comparing the original and counterfactual images concept-based.
    When using concept constraints on the counterfactual generation, we want to check, 
    whether the major change towards the counterfactual is in the selected concepts. 
"""
import os
import sys
import h5py
import numpy as np
import hydra
from omegaconf import DictConfig
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from torchvision import transforms

sys.path.append('./')
from run_ldce_baseline import get_classifier
from src.encode_dataset import compute_layer_attributions
sys.path.append('./ldce')
from ldce.sampling_helpers import normalize
from ldce.data.imagenet_classnames import name_map

# Compute activations / attributions of original and counterfactual

def get_ref_file(cfg, output_dir, attribute='activation'):
    # Get reference dataset
    if cfg.classifier_model.name == 'vgg16_bn':
        if attribute == 'activation':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_acts_cf.h5')
        elif attribute == 'attribution':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_rels_cf.h5')
        elif attribute == 'actattr':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_actattr_cf.h5')
    else:
        if attribute == 'activation':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_{cfg.classifier_model.name}_acts_cf.h5')
        elif attribute == 'attribution':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_{cfg.classifier_model.name}_rels_cf.h5')
        elif attribute == 'actattr':
            acts_base_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_{cfg.classifier_model.name}_actattr_cf.h5')

    return acts_base_file_path

def get_representations(cfg, acts_file_path, attribute='activation'):

    with h5py.File(acts_file_path, 'r', locking=False) as acts_base_file:

        if cfg.concept_layer in acts_base_file:
            print(acts_base_file[cfg.concept_layer].keys())
            ref_acts = acts_base_file[cfg.concept_layer][attribute][:,:]
        else:
            print(acts_base_file.keys())
            ref_acts = acts_base_file['attribution'][cfg.concept_layer][:,:]
        # ref_classes = acts_base_file['label'][:]
        # ref_classes = acts_base_file['prediction'][:]

    return ref_acts

    # Normalization
    if norm:
        norms = np.linalg.norm(ref_acts, axis=1)
        ref_acts = (ref_acts.T / norms).T

    # Get encoding of current datapoint
    if attribute == 'attribution':
        acts_cf_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_{cfg.classifier_model.name}_rels_cf.h5')
    else:
        acts_cf_file_path = os.path.join(output_dir, 'data_representation', f'imagenet_{cfg.classifier_model.name}_acts_cf.h5')
    with h5py.File(acts_cf_file_path, 'r', locking=False) as acts_cf_file:

        cf_acts = acts_cf_file['attribution'][cfg.concept_layer][unique_data_idx,:]
        cf_original_classes = acts_cf_file['prediction'][unique_data_idx]

    if norm:
        cf_norm = np.linalg.norm(cf_acts, axis=1)
        cf_acts = (cf_acts.T / cf_norm).T

def run_concept_comparison(cfg, act_base_file, attribute='activation'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_acts = get_representations(cfg, act_base_file, attribute=attribute)
    print(base_acts.shape)

    # Load model
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    out_size = 256
    transform_list = [
        transforms.Resize((out_size, out_size)),
        transforms.ToTensor()
    ]
    transform = transforms.Compose(transform_list)

    for i in range(8, 1000):

        # load counterfactual
        # img = Image.open(os.path.join(cfg.output_dir, 'bucket_0_10/counterfactual', f'{str(i).zfill(5)}.png'))
        img = Image.open(os.path.join(cfg.output_dir, 'bucket_0_10/counterfactual', f'{str(i).zfill(5)}.png'))
        img = transform(img)

        # img = tf.center_crop(img, 224)
        # img = normalize(img)
        img = img[None].to(device)

        # load pth file
        pth_file = os.path.join(cfg.output_dir, f"bucket_0_10/{str(i).zfill(5)}.pth")
        data = torch.load(pth_file, map_location="cpu")

        conditions = data['conditions']
        print(conditions)

        target = data['target']
        class_target = list(name_map.keys())[list(name_map.values()).index(target)]
        class_target = torch.tensor([class_target], device=device)
        print(class_target)

        orig_img = data['image'][None].to(device)
        print(orig_img.size())

        print(img.size())
        logits = classifier_model(img)
        # print(logits)
        in_class_pred = logits.argmax(dim=1)
        print(in_class_pred)

        b_acts, b_norm_acts, b_attrs, b_norm_attrs, b_rf_neurons = compute_layer_attributions(classifier_model, orig_img, class_target, layers=[cfg.concept_layer])
        acts, norm_acts, attrs, norm_attrs, rf_neurons = compute_layer_attributions(classifier_model, img, class_target, layers=[cfg.concept_layer])
        print(acts.keys())
        print(acts[cfg.concept_layer].shape)

        b_act = b_acts[cfg.concept_layer]
        cf_act = acts[cfg.concept_layer]

        # cf_acts = None

        # Derive concept-based differences
        concept_diff = cf_act[0, conditions] - b_act[0, conditions]
        print(concept_diff)
        # diff = cf_act - base_acts[i]
        # diff = cf_act
        diff = cf_act - b_act

        srt_inds = np.argsort(diff.abs(), axis=1)
        test_srt_inds = np.argsort(diff[0, :].abs())

        assert (srt_inds[0, :] == test_srt_inds).all()

        print(srt_inds[:, -cfg.num_concepts:])
        print(srt_inds[:, :cfg.num_concepts])

        print(diff[0].mean())
        print(diff[0].median())
        print(diff[0, srt_inds[:, -5*cfg.num_concepts:]])

        raise ValueError


    # Rank concepts and compare to concept selection

    raise ValueError

@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:

    attribute = 'activation'       #   'activation'    'attribution
    output_dir = '/results/counterfactuals'

    act_base_file = get_ref_file(cfg, output_dir, attribute=attribute)

    run_concept_comparison(cfg, act_base_file, attribute=attribute)

if __name__ == '__main__':
    main()
