""" Concept-localization evaluation for counterfactuals (cars / boxcars ready).

The point-estimate validity metrics only tell you *whether* the label flipped.
This tells you *whether it flipped for the right reason*: when the sampler is
told to steer a handful of selected concepts (channels) at ``concept_layer``,
did the change actually concentrate in those channels, or did it spread out /
land elsewhere (the tell-tale of an adversarial-looking flip)?

For every sample it compares the per-channel change between the original and the
counterfactual at ``concept_layer`` and reports

    quality_ratio = sum_{c in selected} |delta_c|  /  sum of the top-k |delta_c|

where k = number of selected concepts. quality = 1.0 means the selected
concepts *are* exactly the top-k most-changed channels; the random baseline
(k channels drawn at random) says what an uninformative selection would score.
Both come with bootstrap CIs, and there is an optional per-target breakdown so
you can see where localization is weakest.

This is a portable rewrite of compare_concept_activations.py, which was wired to
1000 ImageNet classes and a hardcoded ``bucket_0_10`` path. Here everything --
class count, layer resolution under the ``Normalizer`` wrapper, bucket layout --
comes from the run's own Hydra config, so it works unchanged on Stanford Cars
and BoxCars116k.

Two metrics (``metric=`` override):

* ``activation`` (default): channel-activation difference. Needs only a forward
  hook, so it is robust to the classifier wrapper and needs no target class.
* ``attribution``: LRP attribution difference (EpsilonGammaBox), matching the
  original script's signal. Needs the target class index, recovered from the
  dataset's class names.

Run it like the generation scripts (Hydra), pointing at an existing output dir::

    python -m src.evaluation.concept_localization \
        --config-name v1_cars_concept \
        output_dir=/results/counterfactuals/cars_vgg16bn_concept

Add ``+metric=attribution`` for the LRP variant, or
``+export_dir=/results/eval/cars_concept_localization`` to dump CSV/JSON.
(``metric``/``export_dir``/``n_random``/``seed``/``alpha`` are not in the base
config, so use Hydra's ``+key=value`` append syntax to set them.)
"""
import os
import csv
import glob
import json
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from run_ldce_baseline import get_classifier
from src.latent_representation.representations import compute_layer_attributions
from src.evaluation.analyze_results import bootstrap_ci, _write_csv, _write_json


def _resolve_layer_name(model, layer_name):
    """Return the actual named_modules key for ``layer_name``.

    Handles the Normalizer(classifier) wrapper, where "features.37" really lives
    at "classifier.features.37" -- same suffix-match convention as
    compute_concept_conditioning().
    """
    for name, _ in model.named_modules():
        if name == layer_name or name.endswith("." + layer_name):
            return name
    raise KeyError(f"concept_layer {layer_name!r} not found in classifier")


def _get_class_names(cfg):
    """Instantiate the dataset just to read its class-name list (no image IO)."""
    dataset = hydra.utils.instantiate(cfg.data, transform=transforms.ToTensor())
    return dataset.get_class_names()


def _channel_reduce(t):
    """(1, C, H, W) -> (C,) by summing spatial dims; (1, C) passes through."""
    if t.dim() == 4:
        return t.sum(dim=(2, 3))[0]
    return t[0]


def _ratios(delta_abs, conditions, rng, n_random=10):
    """quality and random ratio for one sample from a per-channel |delta|."""
    k = len(conditions)
    top_sum = float(np.sort(delta_abs)[-k:].sum())
    if top_sum <= 0:
        return float("nan"), float("nan")
    cond_sum = float(delta_abs[conditions].sum())
    rand = np.mean([delta_abs[rng.choice(len(delta_abs), k, replace=False)].sum()
                    for _ in range(n_random)])
    return cond_sum / top_sum, float(rand) / top_sum


def _load_img(path, size=256):
    img = Image.open(path).convert("RGB")
    return transforms.functional.to_tensor(transforms.functional.resize(img, [size, size]))


def _iter_samples(output_path):
    """Yield (pth_file, original_png, counterfactual_png) for numeric samples."""
    if "dvce" in output_path:
        buckets = [output_path]
    else:
        buckets = sorted(glob.glob(os.path.join(output_path, "bucket*")))
    for bucket in buckets:
        for pth_file in sorted(glob.glob(os.path.join(bucket, "*.pth"))):
            base = os.path.basename(pth_file)[:-4]
            if not base.isdigit():
                continue
            orig = os.path.join(bucket, "original", base + ".png")
            cf = os.path.join(bucket, "counterfactual", base + ".png")
            if os.path.exists(orig) and os.path.exists(cf):
                yield pth_file, orig, cf


def run_concept_localization(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = cfg.get("metric", "activation")
    n_random = int(cfg.get("n_random", 10))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    classifier = get_classifier(cfg, device)
    classifier.to(device).eval()
    layer_name = _resolve_layer_name(classifier, cfg.concept_layer)

    name_to_idx = None
    num_classes = None
    if metric == "attribution":
        class_names = _get_class_names(cfg)
        name_to_idx = {n: i for i, n in enumerate(class_names)}
        num_classes = len(class_names)

    # forward hook for the activation metric
    captured = {}
    if metric == "activation":
        module = dict(classifier.named_modules())[layer_name]

        def _hook(_m, _in, out):
            captured["out"] = out.detach()
        handle = module.register_forward_hook(_hook)

    qualities, randoms, targets = [], [], []
    samples = list(_iter_samples(cfg.output_dir))
    for pth_file, orig_png, cf_png in tqdm(samples, leave=False):
        data = torch.load(pth_file, map_location="cpu", weights_only=False)
        conditions = np.asarray(data["conditions"]).astype(int).ravel()
        orig = _load_img(orig_png)[None].to(device)
        cf = _load_img(cf_png)[None].to(device)

        if metric == "activation":
            with torch.inference_mode():
                classifier(orig)
                a_orig = _channel_reduce(captured["out"].cpu())
                classifier(cf)
                a_cf = _channel_reduce(captured["out"].cpu())
            delta_abs = (a_cf - a_orig).abs().numpy()
        else:  # attribution (LRP)
            tgt = torch.tensor([name_to_idx[data["target"]]], device=device)
            b = compute_layer_attributions(classifier, orig, tgt, layers=[layer_name], num_classes=num_classes)[2]
            c = compute_layer_attributions(classifier, cf, tgt, layers=[layer_name], num_classes=num_classes)[2]
            delta_abs = (c[layer_name] - b[layer_name])[0].abs().numpy()

        q, r = _ratios(delta_abs, conditions, rng, n_random=n_random)
        qualities.append(q)
        randoms.append(r)
        targets.append(data["target"])

    if metric == "activation":
        handle.remove()

    qualities = np.asarray(qualities, dtype=np.float64)
    randoms = np.asarray(randoms, dtype=np.float64)
    q_mean, q_lo, q_hi = bootstrap_ci(qualities, alpha=cfg.get("alpha", 0.05))
    r_mean, r_lo, r_hi = bootstrap_ci(randoms, alpha=cfg.get("alpha", 0.05))

    summary = {
        "output_dir": cfg.output_dir,
        "metric": metric,
        "concept_layer": cfg.concept_layer,
        "resolved_layer": layer_name,
        "num_concepts": int(cfg.get("num_concepts", len(np.atleast_1d(conditions)))),
        "n_samples": int(qualities.size),
        "quality_ratio_mean": q_mean,
        "quality_ratio_ci_lo": q_lo,
        "quality_ratio_ci_hi": q_hi,
        "random_ratio_mean": r_mean,
        "random_ratio_ci_lo": r_lo,
        "random_ratio_ci_hi": r_hi,
    }

    # per-target breakdown (worst localization first)
    per_target = {}
    for q, tgt in zip(qualities, targets):
        per_target.setdefault(tgt, []).append(q)
    per_target_rows = []
    for tgt, qs in per_target.items():
        qs = np.asarray(qs, dtype=np.float64)
        per_target_rows.append({
            "target": tgt,
            "n": int(qs.size),
            "quality_ratio_mean": float(np.nanmean(qs)),
            "quality_ratio_std": float(np.nanstd(qs)),
        })
    per_target_rows.sort(key=lambda x: x["quality_ratio_mean"])

    return summary, per_target_rows, qualities, randoms


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg: DictConfig) -> None:
    summary, per_target_rows, qualities, randoms = run_concept_localization(cfg)

    print(f"\n=== concept localization ({summary['metric']}) : {summary['output_dir']} ===")
    print(f"layer {summary['concept_layer']} -> {summary['resolved_layer']}  "
          f"({summary['num_concepts']} concepts, {summary['n_samples']} samples)")
    print(f"quality ratio: {summary['quality_ratio_mean']:.4f} "
          f"[{summary['quality_ratio_ci_lo']:.4f}, {summary['quality_ratio_ci_hi']:.4f}]")
    print(f"random  ratio: {summary['random_ratio_mean']:.4f} "
          f"[{summary['random_ratio_ci_lo']:.4f}, {summary['random_ratio_ci_hi']:.4f}]")

    print("\nWorst-localized targets (top 10):")
    for row in per_target_rows[:10]:
        print(f"  {row['target'][:40]:40s} n={row['n']:3d}  q={row['quality_ratio_mean']:.4f}")

    export_dir = cfg.get("export_dir", None)
    if export_dir:
        _write_json(os.path.join(export_dir, "concept_localization.json"), summary)
        _write_csv(os.path.join(export_dir, "concept_localization_per_target.csv"), per_target_rows)
        os.makedirs(export_dir, exist_ok=True)
        np.save(os.path.join(export_dir, "quality_ratios.npy"), qualities)
        np.save(os.path.join(export_dir, "random_ratios.npy"), randoms)
        print(f"\nWrote concept_localization.json/.csv + .npy arrays to {export_dir}")


if __name__ == "__main__":
    main()
