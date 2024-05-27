""" Helper functions for datasets and models. """
import sys
import hydra
from hydra.utils import instantiate
import numpy as np
import random
import torch
import torchvision
from torchvision import transforms, datasets

# sys.path.append("./ldce")
from ldce.utils.madry_net import MadryNet
from ldce.utils.preprocessor import Crop, CropAndNormalizer


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# Model helpers
def get_classifier(cfg, device):
    classifier_name = cfg.classifier_model.name
    if "ImageNet" in cfg.data._target_:
        if classifier_name == "robust_resnet50":
            classifier_model = MadryNet(cfg.classifier_model.ckpt, device)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = Crop(classifier_model)
        else:
            classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = CropAndNormalizer(classifier_model)
    elif "Flowers" in cfg.data._target_:
        weights_path = "/results/models/vgg16bn_flowers_20240503_122623_78_0.870"
        classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
        num_ftrs = classifier_model.classifier[6].in_features
        classifier_model.classifier[6] = torch.nn.Linear(num_ftrs, 103)
        classifier_model.load_state_dict(torch.load(weights_path))
        classifier_model.to(device)

    elif "Pets" in cfg.data._target_:
        weights_path = "/results/models/vgg16bn_pets_20240503_092321_7_0.92"
        classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
        num_ftrs = classifier_model.classifier[6].in_features
        classifier_model.classifier[6] = torch.nn.Linear(num_ftrs, 37)
        classifier_model.load_state_dict(torch.load(weights_path))
        classifier_model.to(device)
    else:
        raise NotImplementedError
    return classifier_model

# Data helpers
def get_dataset(cfg, last_data_idx: int = 0, base=False):
    if "ImageNet" in cfg.data._target_:
        out_size = 256
        transform_list = [
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)

        if base:
            dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
        elif 'cf_data' in cfg:
            dataset = instantiate(cfg.cf_data, start_sample=cfg.cf_data.start_sample, end_sample=cfg.cf_data.end_sample, transform=transform, restart_idx=last_data_idx)
        else:
            dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
    # elif "CelebAHQDataset" in cfg.data._target_:
    #     dataset = instantiate(
    #         cfg.data,
    #         image_size=256, 
    #         data_dir=cfg.data.data_dir, 
    #         random_crop=False, 
    #         random_flip=False, 
    #         partition='test',
    #         query_label=cfg.data.query_label,
    #         normalize=False,
    #         shard=cfg.data.shard,
    #         num_shards=cfg.data.num_shards,
    #         restart_idx=last_data_idx
    #     )
    elif "Flowers102" in cfg.data._target_:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    elif "OxfordIIIPets" in cfg.data._target_: # try running on 224x224 img
        def _convert_to_rgb(image):
            return image.convert('RGB')
        out_size = 256
        transform_list = [
            transforms.Resize((out_size, out_size)),
            # transforms.CenterCrop(out_size),
            _convert_to_rgb,
            transforms.ToTensor(),
        ]
        transform = transforms.Compose(transform_list)
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    else:
        raise NotImplementedError
    return dataset