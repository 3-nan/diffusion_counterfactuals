""" Statistical analysis of counterfactual runs.

Adds what the point-estimate scripts (compute_validity_metrics.py,
compute_lpnorms.py, compute_fid.py) lack:

  * confidence intervals on the flip ratio (Wilson score interval -- exact for a
    proportion, no re-run needed) and bootstrap CIs on confidence / L_p;
  * a "when does it fail" breakdown of the flip ratio per target class and per
    (source -> target) pair, each with its own Wilson CI and sorted worst-first;
  * across-seed aggregation (mean +/- std over seeds + a pooled Wilson CI);
  * a concept_layer sensitivity summary (flip ratio +/- CI per layer + spread).

It reads the same per-sample ``*.pth`` dicts the other evaluators consume
(keys: ``target``, ``out_pred``, ``source``, ``out_confid``, ``out_tgt_confid``,
``in_confid``, ``closness_1``, ``closness_2`` -- see run_concept_ldce.py), so it
works on *existing* output directories without regenerating anything.

CLI examples
------------
Single run, print report + dump CSV/JSON::

    python -m src.evaluation.analyze_results single \
        --output-path /results/counterfactuals/cars_vgg16bn_concept \
        --export-dir /results/eval/cars_concept

Aggregate several seed runs::

    python -m src.evaluation.analyze_results seeds \
        --output-paths /results/.../cars_concept_seed0 /results/.../cars_concept_seed1 ... \
        --export-dir /results/eval/cars_concept_seeds

Layer sensitivity (name=path pairs)::

    python -m src.evaluation.analyze_results layers \
        --layer features.34=/results/.../cars_concept_features.34 \
        --layer features.37=/results/.../cars_concept_features.37 \
        --layer features.40=/results/.../cars_concept_features.40 \
        --export-dir /results/eval/cars_layer_sweep
"""
import os
import csv
import glob
import json
import argparse
from statistics import NormalDist
from collections import defaultdict

import numpy as np
import torch

# Scalar keys we care about -- everything else in the dict (image tensors,
# videos, concept vectors) is skipped so loading stays cheap.
_SCALAR_KEYS = (
    "unique_id", "source", "target", "in_pred", "out_pred",
    "out_confid", "out_tgt_confid", "in_confid", "in_tgt_confid",
    "closness_1", "closness_2",
)


def _z(alpha):
    """Two-sided normal quantile for confidence level ``1 - alpha``."""
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_interval(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion k/n.

    Preferred over the normal approximation for the flip ratio because it stays
    inside [0, 1] and behaves at the 82-96% range and small per-class n where
    the failure analysis operates.
    """
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    """Percentile bootstrap CI for a statistic of ``values``."""
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat(values)), float(lo), float(hi)


def load_records(output_path):
    """Load per-sample scalar records from one output directory.

    Handles both layouts used across the repo: the flat ``dvce`` layout
    (``*.pth`` directly under ``output_path``) and the bucketed layout
    (``bucket*/*.pth``). Non-numeric bookkeeping files (config, checkpoints)
    are skipped.
    """
    if "dvce" in output_path:
        pth_files = sorted(glob.glob(os.path.join(output_path, "*.pth")))
    else:
        pth_files = sorted(glob.glob(os.path.join(output_path, "bucket*", "*.pth")))

    records = []
    for pth_file in pth_files:
        if not os.path.basename(pth_file)[:-4].isdigit():
            continue
        data = torch.load(pth_file, map_location="cpu", weights_only=False)
        if "target" not in data or "out_pred" not in data:
            continue
        rec = {}
        for key in _SCALAR_KEYS:
            val = data.get(key)
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except (ValueError, RuntimeError):
                    val = None
            rec[key] = val
        rec["flipped"] = int(rec["target"] == rec["out_pred"])
        records.append(rec)
    return records


def _col(records, key):
    return [r[key] for r in records if r.get(key) is not None]


def summarize(records, alpha=0.05, n_boot=10000, seed=0):
    """Overall metrics with CIs for one set of records."""
    n = len(records)
    k = sum(r["flipped"] for r in records)
    fr, fr_lo, fr_hi = wilson_interval(k, n, alpha)

    def _boot(key):
        return bootstrap_ci(_col(records, key), n_boot=n_boot, alpha=alpha, seed=seed)

    tgt_conf, tc_lo, tc_hi = _boot("out_tgt_confid")
    l1, l1_lo, l1_hi = _boot("closness_1")
    l2, l2_lo, l2_hi = _boot("closness_2")
    return {
        "n_samples": n,
        "n_flipped": k,
        "flip_ratio": fr,
        "flip_ratio_ci_lo": fr_lo,
        "flip_ratio_ci_hi": fr_hi,
        "mean_target_confidence": tgt_conf,
        "target_confidence_ci_lo": tc_lo,
        "target_confidence_ci_hi": tc_hi,
        "mean_closness_1": l1,
        "closness_1_ci_lo": l1_lo,
        "closness_1_ci_hi": l1_hi,
        "mean_closness_2": l2,
        "closness_2_ci_lo": l2_lo,
        "closness_2_ci_hi": l2_hi,
    }


def _group_breakdown(records, key_fn, alpha=0.05):
    groups = defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    rows = []
    for key, recs in groups.items():
        n = len(recs)
        k = sum(r["flipped"] for r in recs)
        fr, lo, hi = wilson_interval(k, n, alpha)
        confs = _col(recs, "out_tgt_confid")
        rows.append({
            "key": key,
            "n": n,
            "n_flipped": k,
            "flip_ratio": fr,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_width": hi - lo,
            "mean_target_confidence": float(np.mean(confs)) if confs else float("nan"),
        })
    # worst-first, then by wider uncertainty
    rows.sort(key=lambda x: (x["flip_ratio"], -x["ci_width"]))
    return rows


def per_target_breakdown(records, alpha=0.05):
    """Flip ratio + Wilson CI per target class (the class we push *towards*)."""
    return _group_breakdown(records, lambda r: r["target"], alpha)


def per_pair_breakdown(records, alpha=0.05):
    """Flip ratio + Wilson CI per (source -> target) pair."""
    rows = _group_breakdown(records, lambda r: (r["source"], r["target"]), alpha)
    for row in rows:
        src, tgt = row.pop("key")
        row["source"] = src
        row["target"] = tgt
    return rows


def aggregate_seeds(output_paths, alpha=0.05, n_boot=10000, seed=0):
    """Aggregate flip ratio across seed runs.

    Reports both the across-seed mean +/- std (variability the method itself
    introduces) and a pooled Wilson CI over all samples (sampling uncertainty).
    """
    per_seed = []
    pooled = []
    for path in output_paths:
        recs = load_records(path)
        if not recs:
            per_seed.append({"path": path, "n_samples": 0, "flip_ratio": float("nan")})
            continue
        n = len(recs)
        k = sum(r["flipped"] for r in recs)
        per_seed.append({
            "path": path,
            "n_samples": n,
            "n_flipped": k,
            "flip_ratio": k / n,
            "mean_target_confidence": float(np.mean(_col(recs, "out_tgt_confid"))),
        })
        pooled.extend(recs)

    frs = [s["flip_ratio"] for s in per_seed if s["n_samples"] > 0]
    n_tot = len(pooled)
    k_tot = sum(r["flipped"] for r in pooled)
    pooled_fr, pooled_lo, pooled_hi = wilson_interval(k_tot, n_tot, alpha)
    summary = {
        "n_seeds": len(frs),
        "flip_ratio_mean": float(np.mean(frs)) if frs else float("nan"),
        "flip_ratio_std": float(np.std(frs, ddof=1)) if len(frs) > 1 else 0.0,
        "flip_ratio_min": float(np.min(frs)) if frs else float("nan"),
        "flip_ratio_max": float(np.max(frs)) if frs else float("nan"),
        "pooled_n_samples": n_tot,
        "pooled_flip_ratio": pooled_fr,
        "pooled_ci_lo": pooled_lo,
        "pooled_ci_hi": pooled_hi,
    }
    return summary, per_seed


def _collect_image_pairs(output_path):
    """Return (real_files, gen_files) as sorted lists, matching the layout logic
    used by compute_fid.py (flat dvce layout vs. bucketed layout)."""
    real, gen = [], []
    if "dvce" in output_path:
        real = sorted(glob.glob(os.path.join(output_path, "original", "*")))
        gen = sorted(glob.glob(os.path.join(output_path, "counterfactual", "*")))
    else:
        for bucket in sorted(glob.glob(os.path.join(output_path, "bucket*"))):
            real.extend(sorted(glob.glob(os.path.join(bucket, "original", "*"))))
            gen.extend(sorted(glob.glob(os.path.join(bucket, "counterfactual", "*"))))
    return real, gen


def _frechet_fast(feats_r, feats_g):
    """Exact Fréchet distance between two Gaussians fit to feature matrices.

    Avoids the O(d^3) matrix square root that scipy.linalg.sqrtm (used by
    pytorch_fid) runs on the 2048x2048 covariance -- prohibitive inside a
    bootstrap loop. Because there are far fewer samples (n ~ 1000) than feature
    dims (d = 2048), the covariances are low rank and

        Tr((Sig_r Sig_g)^{1/2}) = sum(svdvals(Xr_c @ Xg_c.T)) / sqrt((n_r-1)(n_g-1))

    where Xr_c / Xg_c are the mean-centred feature matrices. That SVD is on an
    n_r x n_g matrix (~1000x1000) instead of a 2048x2048 sqrtm, so each bootstrap
    resample is fast while staying numerically exact.
    """
    n_r, n_g = feats_r.shape[0], feats_g.shape[0]
    mu_r, mu_g = feats_r.mean(axis=0), feats_g.mean(axis=0)
    xr = feats_r - mu_r
    xg = feats_g - mu_g
    diff = mu_r - mu_g
    tr_r = float((xr * xr).sum()) / (n_r - 1)
    tr_g = float((xg * xg).sum()) / (n_g - 1)
    # singular values of the cross matrix give sum(sqrt(eig(Sig_r Sig_g)))
    s = np.linalg.svd(xr @ xg.T, compute_uv=False)
    tr_sqrt = float(s.sum()) / np.sqrt((n_r - 1) * (n_g - 1))
    return float(diff @ diff) + tr_r + tr_g - 2.0 * tr_sqrt


def fid_with_ci(output_path, n_boot=200, alpha=0.05, seed=0, batch_size=50,
                dims=2048, device=None):
    """FID point estimate + bootstrap confidence interval.

    A proper bootstrap would resample images and recompute FID -- but that reruns
    Inception on every image each time. Instead we extract Inception features
    *once* per set and bootstrap in feature space: each resample only recomputes
    the mean/covariance and the (exact) Fréchet distance via _frechet_fast.

    Important: FID is *positively biased* at finite sample size, and resampling
    with replacement shrinks the effective number of unique samples (~63%), which
    inflates every bootstrap replicate upward. So the bootstrap *distribution* is
    shifted above the point estimate and a naive percentile interval would not
    even cover it. We therefore use the bootstrap only for the *spread* (its
    standard error is still a valid variance estimate) and report a
    normal-approximation interval centred on the unbiased point estimate:
        FID +/- z * SE_boot.
    Use it to compare runs; it does not remove FID's small-sample bias. For
    between-run uncertainty prefer several generation seeds (see aggregate_seeds).
    """
    import torch as _torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import get_activations

    real_files, gen_files = _collect_image_pairs(output_path)
    if not real_files or not gen_files:
        raise FileNotFoundError(f"no original/counterfactual images under {output_path}")

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    # One forward pass per set -- the expensive part, done exactly once.
    feats_real = get_activations(real_files, model, batch_size, dims, device).astype(np.float64)
    feats_gen = get_activations(gen_files, model, batch_size, dims, device).astype(np.float64)

    point = float(_frechet_fast(feats_real, feats_gen))

    rng = np.random.default_rng(seed)
    n_r, n_g = feats_real.shape[0], feats_gen.shape[0]
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ri = rng.integers(0, n_r, size=n_r)
        gi = rng.integers(0, n_g, size=n_g)
        boot[b] = _frechet_fast(feats_real[ri], feats_gen[gi])
    se = float(boot.std(ddof=1))
    z = _z(alpha)
    lo, hi = point - z * se, point + z * se
    return {
        "fid": point,
        "fid_ci_lo": float(lo),
        "fid_ci_hi": float(hi),
        "fid_se": se,
        "fid_boot_mean": float(boot.mean()),
        "n_real": n_r,
        "n_gen": n_g,
        "n_boot": n_boot,
    }


def _load_run_config(output_path):
    """Load the OmegaConf snapshot each run saves next to its samples.

    run_concept_ldce.py / run_ldce_baseline.py dump the fully-resolved config to
    ``<bucket>/config.yaml`` -- that's what lets us rebuild the *exact* classifier
    (arch, weights, wrapper) a run used, so the embedding distance is measured in
    the same feature space that produced the counterfactual.
    """
    from omegaconf import OmegaConf

    candidates = sorted(glob.glob(os.path.join(output_path, "bucket*", "config.yaml")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(output_path, "config.yaml")))
    if not candidates:
        raise FileNotFoundError(
            f"no config.yaml under {output_path}; needed to rebuild the classifier "
            f"for embedding distances")
    return OmegaConf.load(candidates[0])


def _last_linear_module(model):
    """The final ``nn.Linear`` in the (possibly wrapper-nested) classifier.

    Its *input* is the penultimate embedding fed to the classification head --
    a single, architecture-agnostic definition of "the classifier's embedding"
    that resolves to ResNet's 512-d pooled features, VGG's 4096-d FC activation
    and ViT's 768-d CLS token alike, without per-arch special-casing.
    """
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    return last


def _resolve_module(model, name):
    """Resolve a short layer name (e.g. ``features.37`` / ``layer4.1.conv1`` /
    ``encoder.ln``) to the actual module inside the wrapper-nested classifier.

    get_classifier() wraps the backbone in Normalizer/ResizeAndNormalizer, so the
    live path is ``classifier.<name>`` -- match by suffix the same way the concept
    samplers do, erroring on an ambiguous or missing name rather than guessing.
    """
    matches = [(n, m) for n, m in model.named_modules()
               if n == name or n.endswith("." + name)]
    if not matches:
        raise KeyError(f"concept_layer {name!r} not found in classifier")
    if len(matches) > 1:
        raise KeyError(f"concept_layer {name!r} is ambiguous: {[n for n, _ in matches]}")
    return matches[0][1]


def _pool_layer_output(t):
    """Reduce a raw layer activation to one vector per sample, matching how the
    concept pipeline reads that layer: conv maps ``[B,C,H,W]`` -> spatial mean
    ``[B,C]``; ViT token sequences ``[B,T,C]`` -> token mean ``[B,C]``; already
    flat ``[B,C]`` left as-is."""
    if t.ndim == 4:
        return t.mean(dim=(2, 3))
    if t.ndim == 3:
        return t.mean(dim=1)
    return t


def _pair_by_name(real_files, gen_files):
    """Align original/counterfactual files by basename (both dirs share the
    zero-padded ``NNNNN.png`` naming), keeping only indices present in both."""
    rmap = {os.path.basename(f): f for f in real_files}
    gmap = {os.path.basename(f): f for f in gen_files}
    keys = sorted(set(rmap) & set(gmap))
    return [rmap[k] for k in keys], [gmap[k] for k in keys]


def _extract_embeddings(model, files, linear_layer, concept_layer, batch_size, device):
    """Per-sample embeddings for ``files`` at up to two sites, in one forward pass:

      * ``penultimate`` -- input to the final Linear, via a forward *pre*-hook;
      * ``concept``     -- pooled output of ``concept_layer`` (or ``None`` to skip),
        via a forward hook.

    The model's own preprocessing wrapper (normalize, and resize for ViT) is
    applied exactly as during sampling. Returns a dict of ``[N, d]`` arrays.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    captured = {}

    def pre_hook(_module, inputs):
        captured["penultimate"] = inputs[0].detach()

    def concept_hook(_module, _inputs, output):
        captured["concept"] = _pool_layer_output(output.detach())

    handles = [linear_layer.register_forward_pre_hook(pre_hook)]
    if concept_layer is not None:
        handles.append(concept_layer.register_forward_hook(concept_hook))

    out = {"penultimate": []}
    if concept_layer is not None:
        out["concept"] = []
    try:
        with torch.no_grad():
            for start in range(0, len(files), batch_size):
                batch_files = files[start:start + batch_size]
                imgs = [TF.to_tensor(Image.open(f).convert("RGB")) for f in batch_files]
                x = torch.stack(imgs).to(device)
                model(x)
                out["penultimate"].append(captured["penultimate"].flatten(1).cpu().numpy().astype(np.float64))
                if concept_layer is not None:
                    out["concept"].append(captured["concept"].flatten(1).cpu().numpy().astype(np.float64))
    finally:
        for handle in handles:
            handle.remove()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _paired_distance_stats(emb_real, emb_gen, n_boot, alpha, seed):
    """Mean paired Euclidean and cosine distance (each with a percentile
    bootstrap CI) between aligned original/counterfactual embeddings."""
    diff = emb_real - emb_gen
    l2 = np.sqrt((diff * diff).sum(axis=1))
    norms = np.clip(np.linalg.norm(emb_real, axis=1) * np.linalg.norm(emb_gen, axis=1), 1e-12, None)
    cosine_dist = 1.0 - (emb_real * emb_gen).sum(axis=1) / norms

    l2_mean, l2_lo, l2_hi = bootstrap_ci(l2, n_boot=n_boot, alpha=alpha, seed=seed)
    cos_mean, cos_lo, cos_hi = bootstrap_ci(cosine_dist, n_boot=n_boot, alpha=alpha, seed=seed)
    return {
        "dim": int(emb_real.shape[1]),
        "l2": l2_mean, "l2_ci_lo": l2_lo, "l2_ci_hi": l2_hi,
        "cosine": cos_mean, "cosine_ci_lo": cos_lo, "cosine_ci_hi": cos_hi,
    }


def _channel_change_sparsity(emb_real, emb_gen, n_boot, alpha, seed, taus=(1.0, 2.0)):
    """How *many* channels an intervention moves, rather than by how much.

    Complements the L2/cosine distance (which measures overall displacement) by
    quantifying the sparsity of the per-channel change: if only a handful of
    concept-layer channels move, the edit engaged few concepts; if the change is
    spread over many channels, it engaged many. Each channel's change is first
    standardised by its robust spread (MAD over the originals) so the count is
    scale-free and comparable across channels:

        z_{i,c} = (a^cf_{i,c} - a^orig_{i,c}) / sigma_c ,   sigma_c = 1.4826 * MAD_i(a^orig_{.,c})

    and summarised three ways per sample (each averaged over samples with a
    percentile bootstrap CI):

      * ``n_eff``  -- effective number of changed channels, the participation
        ratio ``||z||_1^2 / ||z||_2^2`` (1 = all change in one channel, C =
        spread evenly over every channel); threshold-free.
      * ``hoyer``  -- Hoyer sparsity in [0, 1] (1 = maximally sparse / one channel).
      * ``l0_tau*``-- literal count of channels moved by more than ``tau`` robust SDs.

    Lower ``n_eff`` / ``l0`` and higher ``hoyer`` mean the counterfactual changed
    fewer concepts.
    """
    diff = emb_gen - emb_real
    # Robust per-channel scale from the originals: ReLU activations are
    # nonnegative and heavy-tailed, so MAD is steadier than std. Fall back to
    # std, then to 1, for dead/constant channels.
    med = np.median(emb_real, axis=0)
    mad = np.median(np.abs(emb_real - med), axis=0) * 1.4826
    scale = np.where(mad > 1e-12, mad, emb_real.std(axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = diff / scale  # [N, C]

    absz = np.abs(z)
    l1 = absz.sum(axis=1)
    sq = (z * z).sum(axis=1)
    l2 = np.sqrt(sq)
    n_eff = (l1 * l1) / np.clip(sq, 1e-12, None)
    C = z.shape[1]
    hoyer = (np.sqrt(C) - l1 / np.clip(l2, 1e-12, None)) / (np.sqrt(C) - 1.0)

    out = {"dim": int(C)}
    for name, vals in (("n_eff", n_eff), ("hoyer", hoyer)):
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed)
        out[name], out[f"{name}_ci_lo"], out[f"{name}_ci_hi"] = m, lo, hi
    for tau in taus:
        k = (absz > tau).sum(axis=1).astype(np.float64)
        m, lo, hi = bootstrap_ci(k, n_boot=n_boot, alpha=alpha, seed=seed)
        key = f"l0_tau{tau:g}"
        out[key], out[f"{key}_ci_lo"], out[f"{key}_ci_hi"] = m, lo, hi
    return out


def embedding_distance_with_ci(output_path, n_boot=10000, alpha=0.05, seed=0,
                               batch_size=50, device=None, concept_layer_override=None):
    """Distance between each original and its counterfactual in the run's own
    classifier, at two feature sites, with bootstrap CIs.

    Unlike FID -- which compares the two image *distributions* through Inception --
    this is a *paired*, per-sample distance in the classifier that was actually
    optimised against, so it directly quantifies how far the counterfactual moved
    the sample in the model's decision-relevant feature space. Being an ordinary
    per-sample statistic (not a plug-in distribution functional like FID), the
    percentile bootstrap over pairs is valid here with no bias caveat.

    Two sites are reported (each with mean L2 and mean cosine distance ``1-cos``):
      * ``penultimate`` -- input to the final Linear (the head-level embedding),
        a fixed, architecture-agnostic reference shared across runs;
      * ``concept``     -- the pooled ``cfg.concept_layer`` activation, i.e. the
        exact space the concept constraint acts in (omitted if the run's config
        has no concept_layer, e.g. baselines).
    """
    from run_ldce_baseline import get_classifier

    cfg = _load_run_config(output_path)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_classifier(cfg, device)
    model.to(device).eval()

    linear_layer = _last_linear_module(model)
    if linear_layer is None:
        raise RuntimeError("could not locate a final nn.Linear in the classifier")

    concept_name = concept_layer_override or cfg.get("concept_layer", None)
    concept_layer = _resolve_module(model, concept_name) if concept_name else None

    real_files, gen_files = _collect_image_pairs(output_path)
    if not real_files or not gen_files:
        raise FileNotFoundError(f"no original/counterfactual images under {output_path}")
    real_files, gen_files = _pair_by_name(real_files, gen_files)
    if not real_files:
        raise FileNotFoundError(f"no matching original/counterfactual basenames under {output_path}")

    emb_real = _extract_embeddings(model, real_files, linear_layer, concept_layer, batch_size, device)
    emb_gen = _extract_embeddings(model, gen_files, linear_layer, concept_layer, batch_size, device)

    result = {
        "classifier": str(cfg.classifier_model.name),
        "concept_layer": str(concept_name) if concept_name else None,
        "n_pairs": int(emb_real["penultimate"].shape[0]),
        "n_boot": n_boot,
        "penultimate": _paired_distance_stats(
            emb_real["penultimate"], emb_gen["penultimate"], n_boot, alpha, seed),
    }
    if concept_layer is not None:
        result["concept"] = _paired_distance_stats(
            emb_real["concept"], emb_gen["concept"], n_boot, alpha, seed)
        result["concept"]["channel_change"] = _channel_change_sparsity(
            emb_real["concept"], emb_gen["concept"], n_boot, alpha, seed)
    return result


def layer_summary(layer_to_path, alpha=0.05):
    """Flip ratio + Wilson CI per concept_layer, plus the spread across layers."""
    rows = []
    for layer, path in layer_to_path.items():
        recs = load_records(path)
        n = len(recs)
        k = sum(r["flipped"] for r in recs)
        fr, lo, hi = wilson_interval(k, n, alpha)
        confs = _col(recs, "out_tgt_confid")
        rows.append({
            "concept_layer": layer,
            "path": path,
            "n_samples": n,
            "n_flipped": k,
            "flip_ratio": fr,
            "ci_lo": lo,
            "ci_hi": hi,
            "mean_target_confidence": float(np.mean(confs)) if confs else float("nan"),
        })
    frs = [r["flip_ratio"] for r in rows if r["n_samples"] > 0]
    sensitivity = {
        "n_layers": len(frs),
        "flip_ratio_mean": float(np.mean(frs)) if frs else float("nan"),
        "flip_ratio_std": float(np.std(frs, ddof=1)) if len(frs) > 1 else 0.0,
        "flip_ratio_range": (float(np.max(frs)) - float(np.min(frs))) if frs else float("nan"),
    }
    rows.sort(key=lambda x: x["flip_ratio"])
    return sensitivity, rows


# --------------------------------------------------------------------------- #
# Export / printing helpers
# --------------------------------------------------------------------------- #
def _write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _fmt_ci(p, lo, hi):
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def _cmd_single(args):
    records = load_records(args.output_path)
    if not records:
        print(f"No records found under {args.output_path}")
        return
    overall = summarize(records, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed)
    by_target = per_target_breakdown(records, alpha=args.alpha)
    by_pair = per_pair_breakdown(records, alpha=args.alpha)

    print(f"\n=== {args.output_path} ===")
    print(f"samples: {overall['n_samples']}  flipped: {overall['n_flipped']}")
    print("flip ratio        :", _fmt_ci(overall["flip_ratio"], overall["flip_ratio_ci_lo"], overall["flip_ratio_ci_hi"]))
    print("target confidence :", _fmt_ci(overall["mean_target_confidence"], overall["target_confidence_ci_lo"], overall["target_confidence_ci_hi"]))
    print("closeness L1      :", _fmt_ci(overall["mean_closness_1"], overall["closness_1_ci_lo"], overall["closness_1_ci_hi"]))
    print("closeness L2      :", _fmt_ci(overall["mean_closness_2"], overall["closness_2_ci_lo"], overall["closness_2_ci_hi"]))

    print(f"\nHardest targets (worst flip ratio, top {args.top}):")
    for row in by_target[: args.top]:
        print(f"  {row['key'][:40]:40s} n={row['n']:3d}  fr={_fmt_ci(row['flip_ratio'], row['ci_lo'], row['ci_hi'])}")

    if args.export_dir:
        _write_json(os.path.join(args.export_dir, "overall.json"), overall)
        _write_csv(os.path.join(args.export_dir, "per_target.csv"), by_target)
        _write_csv(os.path.join(args.export_dir, "per_pair.csv"), by_pair)
        print(f"\nWrote overall.json, per_target.csv, per_pair.csv to {args.export_dir}")


def _cmd_seeds(args):
    summary, per_seed = aggregate_seeds(args.output_paths, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed)
    print("\n=== seed aggregation ===")
    print(f"seeds: {summary['n_seeds']}")
    print(f"flip ratio across seeds: {summary['flip_ratio_mean']:.4f} +/- {summary['flip_ratio_std']:.4f} "
          f"(min {summary['flip_ratio_min']:.4f}, max {summary['flip_ratio_max']:.4f})")
    print(f"pooled flip ratio      : {_fmt_ci(summary['pooled_flip_ratio'], summary['pooled_ci_lo'], summary['pooled_ci_hi'])}")
    for s in per_seed:
        print(f"  {s['path']}  n={s['n_samples']}  fr={s['flip_ratio']:.4f}")
    if args.export_dir:
        _write_json(os.path.join(args.export_dir, "seed_summary.json"), summary)
        _write_csv(os.path.join(args.export_dir, "per_seed.csv"), per_seed)
        print(f"\nWrote seed_summary.json, per_seed.csv to {args.export_dir}")


def _parse_layer_arg(items):
    mapping = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--layer expects name=path, got {item!r}")
        name, path = item.split("=", 1)
        mapping[name] = path
    return mapping


def _cmd_layers(args):
    layer_to_path = _parse_layer_arg(args.layer)
    sensitivity, rows = layer_summary(layer_to_path, alpha=args.alpha)
    print("\n=== concept_layer sensitivity ===")
    print(f"layers: {sensitivity['n_layers']}  flip ratio {sensitivity['flip_ratio_mean']:.4f} "
          f"+/- {sensitivity['flip_ratio_std']:.4f}  (range {sensitivity['flip_ratio_range']:.4f})")
    for row in rows:
        print(f"  {row['concept_layer']:14s} n={row['n_samples']:4d}  "
              f"fr={_fmt_ci(row['flip_ratio'], row['ci_lo'], row['ci_hi'])}")
    if args.export_dir:
        _write_json(os.path.join(args.export_dir, "layer_sensitivity.json"), sensitivity)
        _write_csv(os.path.join(args.export_dir, "per_layer.csv"), rows)
        print(f"\nWrote layer_sensitivity.json, per_layer.csv to {args.export_dir}")


def _cmd_fid(args):
    # Each FID bootstrap resample recomputes an ~n x n SVD (via _frechet_fast),
    # far cheaper than the old 2048x2048 sqrtm but still ~50ms each, so the 10k
    # default (fine for the vectorized proportion/L_p bootstraps) is overkill --
    # fall back to 1000 unless the user set --n-boot explicitly to something smaller.
    n_boot = args.n_boot if args.n_boot < 10000 else 1000
    result = fid_with_ci(args.output_path, n_boot=n_boot, alpha=args.alpha,
                         seed=args.seed, batch_size=args.batch_size, dims=args.dims)
    print(f"\n=== FID  {args.output_path} ===")
    print(f"real: {result['n_real']}  gen: {result['n_gen']}  bootstrap resamples: {result['n_boot']}")
    print("FID:", _fmt_ci(result["fid"], result["fid_ci_lo"], result["fid_ci_hi"]),
          f"(SE {result['fid_se']:.3f}, normal-approx CI centred on point estimate)")
    if args.export_dir:
        _write_json(os.path.join(args.export_dir, "fid.json"), result)
        print(f"\nWrote fid.json to {args.export_dir}")


def _cmd_embed(args):
    result = embedding_distance_with_ci(
        args.output_path, n_boot=args.n_boot, alpha=args.alpha,
        seed=args.seed, batch_size=args.batch_size,
        concept_layer_override=args.concept_layer)
    print(f"\n=== Classifier embedding distance  {args.output_path} ===")
    print(f"classifier: {result['classifier']}  pairs: {result['n_pairs']}")
    pen = result["penultimate"]
    print(f"penultimate (dim {pen['dim']}):")
    print("  L2      :", _fmt_ci(pen["l2"], pen["l2_ci_lo"], pen["l2_ci_hi"]))
    print("  cosine  :", _fmt_ci(pen["cosine"], pen["cosine_ci_lo"], pen["cosine_ci_hi"]))
    if "concept" in result:
        con = result["concept"]
        print(f"concept_layer {result['concept_layer']} (dim {con['dim']}):")
        print("  L2      :", _fmt_ci(con["l2"], con["l2_ci_lo"], con["l2_ci_hi"]))
        print("  cosine  :", _fmt_ci(con["cosine"], con["cosine_ci_lo"], con["cosine_ci_hi"]))
        cc = con.get("channel_change")
        if cc:
            print(f"  channel change (of {cc['dim']} channels):")
            print("    n_eff   :", _fmt_ci(cc["n_eff"], cc["n_eff_ci_lo"], cc["n_eff_ci_hi"]))
            print("    hoyer   :", _fmt_ci(cc["hoyer"], cc["hoyer_ci_lo"], cc["hoyer_ci_hi"]))
            print("    L0 >1sd :", _fmt_ci(cc["l0_tau1"], cc["l0_tau1_ci_lo"], cc["l0_tau1_ci_hi"]))
            print("    L0 >2sd :", _fmt_ci(cc["l0_tau2"], cc["l0_tau2_ci_lo"], cc["l0_tau2_ci_hi"]))
    else:
        print("concept_layer: (none in config -- skipped)")
    if args.export_dir:
        _write_json(os.path.join(args.export_dir, "embed_distance.json"), result)
        print(f"\nWrote embed_distance.json to {args.export_dir}")


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--alpha", type=float, default=0.05, help="1-alpha confidence level (default 0.05 -> 95%%).")
    common.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples for non-proportion CIs.")
    common.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed.")
    common.add_argument("--export-dir", type=str, default=None, help="if set, write CSV/JSON here.")

    parser = argparse.ArgumentParser(
        description="Statistical analysis of counterfactual runs (CIs + failure breakdown).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_single = sub.add_parser("single", parents=[common],
                              help="one run: overall CIs + per-target/per-pair failure breakdown.")
    p_single.add_argument("--output-path", required=True, type=str)
    p_single.add_argument("--top", type=int, default=15, help="how many hardest targets to print.")
    p_single.set_defaults(func=_cmd_single)

    p_seeds = sub.add_parser("seeds", parents=[common], help="aggregate flip ratio across seed runs.")
    p_seeds.add_argument("--output-paths", required=True, nargs="+", type=str)
    p_seeds.set_defaults(func=_cmd_seeds)

    p_layers = sub.add_parser("layers", parents=[common], help="concept_layer sensitivity across runs.")
    p_layers.add_argument("--layer", required=True, action="append",
                          help="name=path, repeatable (e.g. --layer features.37=/path/run).")
    p_layers.set_defaults(func=_cmd_layers)

    p_fid = sub.add_parser("fid", parents=[common], help="FID with a feature-space bootstrap CI.")
    p_fid.add_argument("--output-path", required=True, type=str)
    p_fid.add_argument("--batch-size", type=int, default=50)
    p_fid.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p_fid.set_defaults(func=_cmd_fid)

    p_embed = sub.add_parser("embed", parents=[common],
                             help="distance (L2 + cosine) between original & counterfactual in the run's own classifier embedding.")
    p_embed.add_argument("--output-path", required=True, type=str)
    p_embed.add_argument("--batch-size", type=int, default=50)
    p_embed.add_argument("--concept-layer", type=str, default=None,
                         help="override concept layer (e.g. for baselines whose config "
                              "has no concept_layer, pass the concept-based setting's layer).")
    p_embed.set_defaults(func=_cmd_embed)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
