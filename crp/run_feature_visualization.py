"""Compute CRP feature-visualization statistics: for every channel/concept in
every Conv2d/Linear layer, cache which samples maximally activate it (and
which get maximum relevance per class), so visualize_concepts.py's
get_max_reference()/get_stats_reference() can look them up later.

Hydra-driven like the rest of the pipeline -- pass --config-name=v1_cars,
v1_boxcars, v1_pets, v1_flowers, v1_cub, etc. Reuses get_classifier/
get_dataset from run_ldce_baseline.py instead of duplicating per-dataset
model-loading/weights-path logic here.

Runs once per dataset+classifier; heavy (a full forward+backward pass per
sample), so shard across GPUs with cfg.data.shard/num_shards like the
generation scripts do.
"""
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")

import hydra
import torch
import torchvision.transforms as T
import zennit
from omegaconf import DictConfig, OmegaConf
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from zennit.torchvision import ResNetCanonizer

from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization

from run_ldce_baseline import get_classifier, get_dataset, dataset_tag


@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    model_name = cfg.classifier_model.name
    if model_name.startswith("vgg"):
        # canonizers is keyword-only as of zennit>=0.5 (positional in the
        # zennit<=0.4.6 that zennit-crp's metadata still pins) -- pass it
        # explicitly so this doesn't silently bind to the epsilon arg instead.
        composite = EpsilonPlusFlat(canonizers=[SequentialMergeBatchNorm()])
    elif model_name.startswith("resnet"):
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    elif model_name.startswith("vit"):
        layer_map_rules = [
            (zennit.types.Activation, zennit.rules.Pass()),  # ignore activations
            (zennit.types.AvgPool, zennit.rules.Norm()),  # normalize relevance for any AvgPool
            (zennit.types.Convolution, zennit.rules.Epsilon(epsilon=1e-6)),
            (zennit.types.Linear, zennit.rules.Epsilon(epsilon=1e-6)),
        ]
        composite = zennit.composites.LayerMapComposite(layer_map=layer_map_rules)
    else:
        raise ValueError(f"please specify canonizers and composite for '{model_name}'")

    cc = ChannelConcept()
    layer_names = get_layer_names(classifier_model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}

    attribution = CondAttribution(classifier_model)

    # FeatureVisualization needs (img, label) pairs, not the (img, label,
    # index) triples get_dataset()'s datasets return by default -- override
    # return_index/return_tgt_cls the same way visualize_concepts.py's
    # ref_cfg_dict does.
    fv_cfg_dict = {"data": dict(cfg["data"])}
    if "ImageNet" in cfg.data._target_:
        fv_cfg_dict["data"].update({"return_tgt_cls": False})
    else:
        fv_cfg_dict["data"].update({"return_index": False})
    if dataset_tag(cfg) in ("cars", "boxcars", "cub"):
        # Reference samples should come from what actually shaped the
        # concept -- training data -- even though cfg.data.split is 'test'
        # in these yamls (shared with run_ldce_baseline.py/run_concept_ldce.py,
        # which generate counterfactuals against the test set and must stay
        # on it). Override only this local copy, not the shared config.
        # ImageNet's dataset class only implements split='val' (asserted at
        # construction), and Flowers102/OxfordIIIPets hardcode split="test"
        # internally regardless of what's passed -- skip the override there.
        fv_cfg_dict["data"]["split"] = "train"
    fv_cfg = OmegaConf.create(fv_cfg_dict)
    dataset = get_dataset(fv_cfg, last_data_idx=0)
    print(f"dataset length: {len(dataset)}")

    # get_classifier() already wraps the StanfordCars/BoxCars116k classifiers
    # in Normalizer (ImageNet-normalize, no crop, see run_ldce_baseline.py's
    # comments), so normalizing again here would double-normalize -- only
    # the raw ImageNet/Flowers/Pets/CUB models need it applied externally.
    if dataset_tag(cfg) in ("cars", "boxcars"):
        preprocessing = None
    else:
        preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    fv_path = f"/results/counterfactuals/fv_{dataset_tag(cfg)}_{model_name}"
    print(f"writing feature-visualization stats to {fv_path}")

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

    batch_size = cfg.get("fv_batch_size", 32)
    fv.run(composite, 0, len(dataset), batch_size=batch_size)


if __name__ == "__main__":
    main()
