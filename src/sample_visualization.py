""" Run Encoding and Concept Analysis for new samples. """
import os
import sys
import hydra
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm
import zennit

sys.path.append('./ldce')

from crp import get_concept_attribution
from helpers.data_model_helpers import get_dataset, get_classifier
from encode_dataset import append_attributions_to_attribution_database
from representations import compute_layer_attributions


@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg : DictConfig):

    concept_image_path = os.path.join(cfg.output_dir, 'concept_imgs')
    os.makedirs(concept_image_path, exist_ok=True)

    results_path = os.path.join(cfg.output_dir, "data_representation")
    os.makedirs(results_path, exist_ok=True)

    activations_file_path = os.path.join(results_path, 'imagenet_acts_samples.h5')
    attributions_file_path = os.path.join(results_path, 'imagenet_rels_samples.h5')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_k = 10
    layer_names = ['features.15',
                   'features.18',
                   'features.20',
                   'features.22',
                   'features.25',
                   'features.27',
                   'features.29']
    
    # batch_size = cfg.data.batch_size
    batch_size = 10

    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    dataset = get_dataset(cfg)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)


    for i, batch in enumerate(tqdm(data_loader)):

        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            tgt_classes = tgt_classes.to(device) #squeeze()

        image = image.to(device)

        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(image)
            else:
                # if "classifier_wrapper" in cfg.classifier_model: # only works for ImageNet!
                #     x = tf.center_crop(x, 224)
                #     x = normalize(x)
                logits = classifier_model(image)
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)

        # get layer representation
        acts, norm_acts, attrs, norm_attrs, rf_neurons = compute_layer_attributions(classifier_model, image, in_class_pred, layers=layer_names)

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

        for layer in layer_names:

            os.makedirs(os.path.join(concept_image_path, layer), exist_ok=True)

            for l, (img, lab, attr) in enumerate(zip(image, in_class_pred, norm_attrs[layer])):
                # Compute concept explanations for top_k concepts
                cids = torch.argsort(attr)[-top_k:]

                for cid in cids:
                    concept_attr = get_concept_attribution(classifier_model, img.unsqueeze(0), lab.unsqueeze(0), layer, cid)
                    cattr = concept_attr[0].sum(0)
                    expl = zennit.image.imgify(cattr, cmap='coldnhot', symmetric=True)
                    expl.save(os.path.join(concept_image_path, layer, f'{str((i*batch_size)+l).zfill(5)}_{cid}.png'))


        if i > 7:
            raise ValueError

    # Retrieve concept maximizations for data_base
    # out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")


if __name__ == '__main__':
    main()