import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
from typing import Any, Dict, Iterable, Tuple
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, open_dict, OmegaConf
import torch
import torchvision.transforms as T
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from crp.image import plot_grid, vis_opaque_img, imgify
from ldce.data.imagenet_classnames import name_map

from src.helpers.data_model_helpers import get_classifier, get_dataset
from src.sampling_helpers import disabled_train

def load_fv(dataset, model):

    # Prepare feature visualization
    fv_path = '/results/counterfactuals/fv_imagenet_vgg16bn'

    attribution = CondAttribution(model)
    # canonizers = [SequentialMergeBatchNorm()]
    # composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer : cc for layer in layer_names}

    preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

    return fv

def plot_grid(ref_c: Dict[int, Any], cmap_dim=1, cmap="bwr", vmin=None, vmax=None, symmetric=True, resize=None, padding=True, figsize=(6, 6)):
    """
    Plots dictionary of reference images as they are returned of the 'get_max_reference' method. To every element in the list crp.imgify is applied with its respective argument values.

    Parameters:
    ----------
    ref_c: dict with keys: integer and value: several lists filled with torch.Tensor, np.ndarray or PIL Image
        To every element in the list crp.imgify is applied.
    resize: None or int
        If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
    padding: boolean
        If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
    figsize: tuple or None
        Size of plt.figure
    cmap_dim: int, 0 or 1 
        Applies the remaining parameters to the first or second element of the tuple list, i.e. plot as heatmap

    REMAINING PARAMETERS: correspond to zennit.imgify

    Returns:
    --------
    shows matplotlib.pyplot plot
    """

    keys = list(ref_c.keys())
    nrows = len(keys)
    value = next(iter(ref_c.values()))

    if cmap_dim > 2 or cmap_dim < 1 or cmap_dim == None:
        raise ValueError("'cmap_dim' must be 0 or 1 or None.")

    if isinstance(value, Tuple) and isinstance(value[0], Iterable):
        nsubrows = len(value)
        ncols = len(value[0])
    elif isinstance(value, Iterable):
        nsubrows = 1
        ncols = len(value)
    else:
        raise ValueError("'ref_c' dictionary must contain an iterable of torch.Tensor, np.ndarray or PIL Image or a tuple of thereof.")

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, 1, wspace=0, hspace=0.2)

    for i in range(nrows):
        inner = gridspec.GridSpecFromSubplotSpec(nsubrows, ncols, subplot_spec=outer[i], wspace=0, hspace=0.1)

        for sr in range(nsubrows):

            if nsubrows > 1:
                img_list = ref_c[keys[i]][sr]
            else:
                img_list = ref_c[keys[i]]
            
            for c in range(ncols):
                ax = plt.Subplot(fig, inner[sr, c])

                if sr == cmap_dim:
                    img = imgify(img_list[c], cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)
                else:
                    img = imgify(img_list[c], resize=resize, padding=padding)

                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if sr == 0 and c == 0:
                    ax.set_ylabel(keys[i])

                fig.add_subplot(ax)
                
    outer.tight_layout(fig)  
    fig.show()

def show_concepts(i, concept_layer, conditioning_file, fv, i2h):

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)

    with h5py.File(conditioning_file, 'r', locking=False) as cfile:

        image = cfile['image'][i]
        label = cfile['label'][i]
        cf_label = cfile['cf_label'][i]

        print(f'{str(i).zfill(5)}: {label} - {i2h[label]} --> {cf_label} - {i2h[cf_label]}')

        for cond_option in ['sumabs', 'sumequal', 'absmean', 'abssum', 'absmax']:
            conditions = cfile[concept_layer][cond_option]['concepts'][i].astype('int32')
            concept_diff = cfile[concept_layer][cond_option]['diffs'][i]

            ref_c = fv.get_max_reference(conditions, concept_layer, 'relevance', (0, 8), composite=composite, plot_fn=None)
            ref_t_all = {}
            for concept, c_diff in zip(conditions, concept_diff):
                if c_diff > 0:
                    ref_t = fv.get_stats_reference(concept, concept_layer, [cf_label], 'relevance', (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)
                else:
                    ref_t = fv.get_stats_reference(concept, concept_layer, [label], 'relevance', (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)
                ref_t_all.update(ref_t)

            plot_grid(ref_t_all, figsize=(6, 9), padding=False)
            plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'{str(i).zfill(5)}_concepts_{cond_option}.png'))
            plt.close()


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    if "verbose" not in cfg:
        with open_dict(cfg):
            cfg.verbose = True
    if "record_intermediate_results" not in cfg:
        with open_dict(cfg):
            cfg.record_intermediate_results = True

    # if "verbose" in cfg and not cfg.verbose:
    #     blockPrint()

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.chmod(cfg.output_dir, 0o777)
    if "ImageNet" in cfg.data._target_:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    else:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
    # elif "CelebAHQDataset" in cfg.data._target_:
    #     # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
    #     # query label 39 (age): label=0 <-> old and label=1 <-> young
    #     assert cfg.data.query_label in [31, 39]
    #     if 31 == cfg.data.query_label:
    #         i2h = ["no smile", "smile"]
    #     elif 39 == cfg.data.query_label:
    #         i2h = ["old", "young"]
    #     else:
    #         raise NotImplementedError
    # elif "Flowers102" in cfg.data._target_:
    #     with open("data/flowers_idx_to_label.json", "r") as f:
    #         flowers_idx_to_classname = json.load(f)
    #     flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
    #     i2h = flowers_idx_to_classname
    # elif "OxfordIIIPets" in cfg.data._target_:
    #     with open("data/pets_idx_to_label.json", "r") as f:
    #         pets_idx_to_classname = json.load(f)
    #     i2h = {int(k): v for k, v in pets_idx_to_classname.items()}
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    classifier_model.train = disabled_train

    ref_cfg_dict = {
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
    ref_cfg = OmegaConf.create(ref_cfg_dict)
    ref_dataset = get_dataset(ref_cfg)

    concept_layer = 'features.40'

    # Read h5py file
    conditioning_file = os.path.join(out_dir, f'conditioning_{cfg.classifier_model.name}.h5')

    fv = load_fv(ref_dataset, classifier_model)

    for i in range(25):

        show_concepts(i, concept_layer, conditioning_file, fv, i2h)


if __name__ == '__main__':
    main()
