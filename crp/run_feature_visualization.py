import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
import torch
import torchvision
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.vgg import vgg16_bn, vgg16
from torchvision.models import vit_b_16
import torchvision.transforms as T
from PIL import Image
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer
from hydra.utils import instantiate
from omegaconf import OmegaConf

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization

def get_dataset(cfg, last_data_idx: int = 0):
    if "ImageNet" in cfg.data._target_:
        out_size = 256
        transform_list = [
            T.Resize((out_size, out_size)),
            T.ToTensor()
        ]
        transform = T.Compose(transform_list)
        dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
    elif "Flowers102" in cfg.data._target_:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
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
            T.Resize((out_size, out_size)),
            # transforms.CenterCrop(out_size),
            _convert_to_rgb,
            T.ToTensor(),
        ]
        transform = T.Compose(transform_list)
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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# data_name = "imagenet"
data_name = "flowers"
# data_name = "pets"

model_name = "vgg16_bn"
# model_name = "resnet18"

if model_name == "vgg16":
    model = vgg16(True).to(device)
elif model_name == "vgg16_bn":
    model = vgg16_bn(True).to(device)
elif model_name == "resnet18":
    model = resnet18(True).to(device)
elif model_name == "resnet50":
    model = resnet50(True).to(device)
elif model_name == "vit_b_16":
    model = vit_b_16(True).to(device)
else:
    raise ValueError(f'model_name {model_name} not in list, please verify!')

if data_name == "flowers":
    weights_path = "/results/models/vgg16bn_flowers_20240503_122623_78_0.870"
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, 103)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

elif data_name == "pets":
    weights_path = "/results/models/vgg16bn_pets_20240503_092321_7_0.92"
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, 37)
    model.load_state_dict(torch.load(weights_path))
    model.to(device)

model.eval()

if model_name.startswith('vgg'):
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)
elif model_name.startswith('resnet'):
    canonizers = [ResNetCanonizer()]
    composite = EpsilonPlusFlat(canonizers)
else:
    raise ValueError('please specify canonizers and composite')

# Concept definition
cc = ChannelConcept()

layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
layer_map = {layer : cc for layer in layer_names}

attribution = CondAttribution(model)

# separate normalization from resizing for plotting purposes later
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# data_path = '/Data/imagenet/val'
    
# apply no normalization here!
# imagenet_data = torchvision.datasets.ImageNet(data_path, transform=transform, split="val")

# out_size = 256
# transform_list = [
#     T.Resize((out_size, out_size)),
#     T.ToTensor()
# ]
# transform = T.Compose(transform_list)
# # dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
# dataset = torchvision.datasets.ImageNet(data_path, transform=transform, split="val", start_sample=10, end_sample=50)

if data_name == "imagenet":
    cfg_dict = {
        'data': {
            '_target_': 'data.datasets.ImageNet',
            'root': '/Data/imagenet/val',
            'idx_to_tgt_cls_path': './ldce/data/image_idx_to_tgt.yaml',
            'split': 'val',
            'return_tgt_cls': False,
            # 'batch_size': 4
            'start_sample': 10,
            'end_sample': 50
            }
    }
    fv_path = f"/results/counterfactuals/fv_imagenet_{model_name}"
elif data_name == "pets":
    cfg_dict = {
        'data': {
            '_target_': 'data.datasets.OxfordIIIPets',
            'root': '/Data/PETS',
            'return_tgt_cls': False,
            'return_index': False,
            'shard': 0,
            'num_shards': 7
        }
    }
    fv_path = f"/results/counterfactuals/fv_pets_{model_name}"
elif data_name == "flowers":
    cfg_dict = {
        'data': {
            '_target_': 'data.datasets.Flowers102',
            'root': "/Data/flowers",
            'return_tgt_cls': False,
            'return_index': False,
            'shard': 0,
            'num_shards': 10,
        }
#   'batch_size': 4
    }
    fv_path = f"/results/counterfactuals/fv_flowers_{model_name}"

cfg = OmegaConf.create(cfg_dict)
dataset = get_dataset(cfg)

fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

# it will take approximately 20 min on a Titan RTX
saved_files = fv.run(composite, 0, len(dataset), 32, 100)
