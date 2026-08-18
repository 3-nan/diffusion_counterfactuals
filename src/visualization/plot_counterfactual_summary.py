"""One figure per sample: original image, counterfactual image, source/target
class with probabilities, and the concepts that most changed between them,
each illustrated with its own reference-image strip.

Consumes:
  - run_concept_ldce.py's saved per-sample .pth (source/target/predictions/
    confidences/conditions/concept_diff), in cfg.output_dir's bucket dir.
  - crp/run_feature_visualization.py's cached reference-sample stats, in
    /results/counterfactuals/fv_<dataset_tag>_<classifier_name>/.

Both of the above must already exist for the config you pass in -- this
script only assembles the figure, it doesn't run generation or the CRP
analysis pass itself.

Usage (same configs as the rest of the concept pipeline):
    python src/visualization/plot_counterfactual_summary.py --config-name=v1_cars_concept
    python src/visualization/plot_counterfactual_summary.py --config-name=v1_cars_concept +uidx=[3,17,42]
    python src/visualization/plot_counterfactual_summary.py --config-name=v1_cars_concept +num_summary_samples=5
"""
import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import zennit.image
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from crp.image import vis_opaque_img

from ldce.data.imagenet_classnames import name_map

from src.sampling_helpers import disabled_train
from src.concept_conditioning import compute_concept_conditioning
from run_ldce_baseline import get_classifier, get_dataset, dataset_tag


def localize_concepts(cfg, classifier_model, device, image_chw, target_idx, concept_layer, num_classes):
    """Per-channel spatial gradient maps, in the same layer/gradient-direction
    convention run_concept_ldce.py itself used to pick these concepts
    (gradient of the target-class logit w.r.t. concept_layer, evaluated on
    the original image).

    Returns the raw [num_channels, H, W] gradient (not thresholded) --
    localization_heatmap() below turns a single channel's map into a
    continuous, signed heatmap.
    """
    target_tensor = torch.tensor([target_idx], device=device)
    batch = image_chw.unsqueeze(0).to(device)
    _, _, _, grad = compute_concept_conditioning(
        classifier_model, batch, target_tensor, concept_layer,
        num_concepts=cfg.num_concepts, num_classes=num_classes, spatial=True,
        spatial_th=cfg.get("localization_spatial_th", 0.4), cond_option=cfg.cond_option,
        classifier_wrapper=cfg.classifier_model.get("classifier_wrapper", False),
        return_gradient=True,
    )
    return torch.from_numpy(grad[0])


def localization_heatmap(image_hwc, channel_grad, diff, spatial_th, cmap="inferno"):
    """Continuous, signed heatmap of one concept channel's spatial gradient,
    alpha-blended over the (already letterbox-cropped) image.

    Signed the same way as before (channel_grad if diff > 0 else -channel_grad,
    matching whether this concept is being added or removed) so only the
    direction-consistent side of the map can show at all -- a "remove this"
    channel's locally counteracting positive sub-region stays fully
    transparent rather than being highlighted. Unlike a hard threshold, the
    remaining direction-consistent values ramp continuously from transparent
    at spatial_th (used as an opacity floor, not a cutoff) to fully opaque at
    the channel's own peak, so relative strength within the highlighted
    region is visible instead of a flat in/out contour.
    """
    size = image_hwc.shape[0]
    signed = channel_grad if diff > 0 else -channel_grad
    signed = torch.nn.functional.interpolate(
        signed[None, None].float(), size=size, mode="bilinear"
    )[0, 0].clamp(min=0)

    peak = signed.max()
    if peak > 0:
        norm = ((signed / peak) - spatial_th).clamp(min=0) / max(1e-6, 1 - spatial_th)
    else:
        norm = signed
    norm = norm.numpy()

    heat = plt.get_cmap(cmap)(norm)[..., :3]
    alpha = (norm * 0.85)[..., None]
    return image_hwc.numpy() * (1 - alpha) + heat * alpha


def lrp_class_heatmap(fv, composite, image_chw, class_idx, rows, cols, device):
    """Full input-space LRP relevance for a single class logit, rendered as a
    signed red/blue heatmap (zennit's 'coldnhot'). Uses the same
    EpsilonPlusFlat composite as the concept analysis, initialising relevance
    at the class output via conditions=[{"y": [class_idx]}] on the CRP
    CondAttribution object. Cropped with the original image's letterbox
    bounds so it stays pixel-aligned with the displayed image."""
    batch = image_chw.unsqueeze(0).to(device).clone().requires_grad_(True)
    attr = fv.attribution(batch, [{"y": [int(class_idx)]}], composite)
    rel = attr.heatmap[0].detach().cpu()[rows, cols]
    pil = zennit.image.imgify(rel, symmetric=True, cmap="coldnhot")
    return np.asarray(pil.convert("RGB"))


def ssim_diff_heatmap(orig_hwc, cf_hwc, cmap="viridis", win_size=11):
    """1 - local SSIM between original and counterfactual, averaged over
    color channels -- a structural-dissimilarity map that needs no
    pretrained network at all (unlike the VGG-feature attempt, reverted
    after coming out too diffuse/blurry), and is less sensitive to
    low-level img2img/VAE reconstruction noise across the whole canvas than
    a raw |pixel diff| (which just traced the whole vehicle outline), since
    SSIM's local variance/covariance terms respond to genuine structural
    change rather than uniform per-pixel intensity drift.

    Reimplements the standard single-scale SSIM formula directly (windowed
    Gaussian-blur means/variances/covariance via torchvision, not the
    already-installed pytorch-msssim package) because pytorch_msssim's
    public ssim()/ms_ssim() only expose the final spatially-*averaged*
    score -- the per-pixel ssim_map is computed internally but never
    returned, and reaching it means depending on the package's private
    _ssim() internals instead.
    """
    x = orig_hwc.permute(2, 0, 1).unsqueeze(0)
    y = cf_hwc.permute(2, 0, 1).unsqueeze(0)

    def blur(t):
        return TF.gaussian_blur(t, win_size)

    mu_x, mu_y = blur(x), blur(y)
    sigma_x = blur(x * x) - mu_x ** 2
    sigma_y = blur(y * y) - mu_y ** 2
    sigma_xy = blur(x * y) - mu_x * mu_y

    C1, C2 = 0.01 ** 2, 0.03 ** 2  # K1, K2 defaults, data_range=1.0 (images are already [0,1])
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    dissim = (1 - ssim_map.mean(dim=1))[0].clamp(min=0)
    peak = dissim.max()
    norm = (dissim / peak).numpy() if peak > 0 else dissim.numpy()
    return plt.get_cmap(cmap)(norm)[..., :3]


def pack_thumbnails(imgs, height=200):
    """Resize each reference thumbnail to a common height (preserving its
    own aspect ratio) and concatenate them left-to-right with no gap,
    instead of each sitting alone, aspect-padded, inside its own equal-width
    cell -- that per-cell padding was scattering as whitespace between every
    pair of thumbnails. The composite is then shown left-anchored in a
    single wide axis (see plot_summary), so whatever space it doesn't fill
    collects into one block on the right instead, ahead of the localization
    columns."""
    resized = []
    for img in imgs:
        w, h = img.size
        new_w = max(1, round(w * height / h))
        resized.append(np.asarray(img.resize((new_w, height))))
    return np.concatenate(resized, axis=1)


def crop_letterbox_bounds(image_hwc, fill=114 / 255, tol=0.03):
    """Row/column bounds that trim letterbox_resize()'s (data/datasets.py)
    flat-gray padding -- it squares the image without stretching or cropping
    real content by padding the shorter axis with solid (114/255) gray,
    which carries no information, so trim it for display. Computed once from
    the original image (guaranteed exactly flat in the padding) and reused
    for the counterfactual/diff maps too, since the img2img canvas is
    pixel-aligned across all of them and the generated image's padding
    region isn't guaranteed to stay perfectly flat the way the source is.
    """
    is_pad = (image_hwc - fill).abs().amax(dim=-1) < tol
    rows = (~is_pad.all(dim=1)).nonzero(as_tuple=True)[0]
    cols = (~is_pad.all(dim=0)).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return slice(None), slice(None)
    return slice(rows.min().item(), rows.max().item() + 1), slice(cols.min().item(), cols.max().item() + 1)


def resolve_layer(model, layer_name: str) -> str:
    """Map a plain layer name (e.g. "features.37") onto the actual module
    path the fv stats on disk are keyed by. StanfordCars/BoxCars116k wrap
    the classifier in Normalizer (see get_classifier() in run_ldce_baseline.py),
    nesting everything under "classifier." -- crp/run_feature_visualization.py
    ran get_layer_names() against that wrapped model, so a plain "features.37"
    from the yaml won't match what's on disk unless resolved the same way
    compute_concept_conditioning() already does elsewhere in this repo."""
    for name, _ in model.named_modules():
        if name == layer_name or name.endswith("." + layer_name):
            return name
    raise ValueError(f"layer '{layer_name}' not found in model")


def get_class_names(cfg, ref_dataset):
    if "ImageNet" in cfg.data._target_:
        return name_map
    if "Flowers102" in cfg.data._target_:
        import json
        with open("data/flowers_idx_to_label.json", "r") as f:
            raw = json.load(f)
        return {int(k) - 1: v for k, v in raw.items()}
    if "OxfordIIIPets" in cfg.data._target_:
        import json
        with open("data/pets_idx_to_label.json", "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    # CUB / StanfordCars / BoxCars116k all expose get_class_names()
    return {i: v for i, v in enumerate(ref_dataset.get_class_names())}


def build_feature_visualization(cfg, classifier_model):
    if cfg.classifier_model.name == 'vgg16_bn' and "ImageNet" in cfg.data._target_:
        fv_path = '/results/counterfactuals/fv_imagenet_vgg16bn'  # legacy path
    else:
        fv_path = f'/results/counterfactuals/fv_{dataset_tag(cfg)}_{cfg.classifier_model.name}'

    model_name = cfg.classifier_model.name
    if model_name.startswith("vgg"):
        composite = EpsilonPlusFlat(canonizers=[SequentialMergeBatchNorm()])
    else:
        raise NotImplementedError(f"add a canonizer/composite for '{model_name}'")

    cc = ChannelConcept()
    layer_names = get_layer_names(classifier_model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer: cc for layer in layer_names}
    attribution = CondAttribution(classifier_model)

    if dataset_tag(cfg) in ("cars", "boxcars"):
        # classifier already normalizes internally (Normalizer wrapper) --
        # see the matching comment in crp/run_feature_visualization.py.
        preprocessing = None
    else:
        preprocessing = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ref_cfg_dict = {"data": dict(cfg["data"])}
    if "ImageNet" in cfg.data._target_:
        ref_cfg_dict["data"].update({"return_tgt_cls": False})
    else:
        ref_cfg_dict["data"].update({"return_index": False})
    if dataset_tag(cfg) in ("cars", "boxcars", "cub"):
        # Must match crp/run_feature_visualization.py's split override: the
        # fv stats were computed against the training set, not cfg.data.split
        # ('test', used for generation) -- get_max_reference/get_stats_reference
        # index into that stats database by sample index, so this reference
        # dataset has to come from the exact same split.
        ref_cfg_dict["data"]["split"] = "train"
    ref_cfg = OmegaConf.create(ref_cfg_dict)
    ref_dataset = get_dataset(ref_cfg)

    fv = FeatureVisualization(attribution, ref_dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)
    return fv, composite, ref_dataset


def out_dir_for(cfg):
    if "ImageNet" in cfg.data._target_:
        return os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    return os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")


def plot_summary(cfg, classifier_model, device, fv, composite, concept_layer, i2h, data_dict, uidx, save_path):
    conditions = np.asarray(data_dict["conditions"])
    concept_diff = np.asarray(data_dict["concept_diff"])

    n_concepts = min(cfg.get("num_summary_concepts", 6), len(conditions))
    order = np.argsort(-np.abs(concept_diff))[:n_concepts]
    conditions, concept_diff = conditions[order], concept_diff[order]

    source, target = data_dict["source"], data_dict["target"]
    name_to_idx = {v: k for k, v in i2h.items()}
    source_idx, target_idx = name_to_idx[source], name_to_idx[target]

    # Crop letterbox padding for display, using bounds from the original
    # image (guaranteed exactly flat there) for both images and the diff map
    # -- they're pixel-aligned, so one set of bounds applies to all three.
    orig_full = data_dict["image"].clamp(0, 1).permute(1, 2, 0)
    cf_full = data_dict["gen_image"].clamp(0, 1).permute(1, 2, 0)
    rows, cols = crop_letterbox_bounds(orig_full)
    orig_hwc = orig_full[rows, cols]
    cf_hwc = cf_full[rows, cols]
    diff_map = ssim_diff_heatmap(orig_hwc, cf_hwc)

    # Raw per-channel spatial gradient, computed once per sample for each of
    # the original and counterfactual images (gradient of the target-class
    # logit w.r.t. concept_layer -- same direction run_concept_ldce.py used
    # to pick these concepts in the first place). Turned into a heatmap per
    # concept below, signed to match that concept's own add/remove
    # direction. Showing both sides lets you check whether the region
    # flagged in the original actually is where the edit landed in the
    # counterfactual.
    spatial_grad = localize_concepts(
        cfg, classifier_model, device, data_dict["image"], target_idx, concept_layer, len(i2h)
    )
    spatial_grad_cf = localize_concepts(
        cfg, classifier_model, device, data_dict["gen_image"], target_idx, concept_layer, len(i2h)
    )
    spatial_th = cfg.get("localization_spatial_th", 0.4)

    add_color, remove_color = "darkgreen", "tab:red"

    refs_per_concept = cfg.get("num_summary_refs", 4)
    attr = "relevance"
    ref_rows = []
    for concept, diff in zip(conditions, concept_diff):
        # A positive diff means this concept increased towards the target --
        # illustrate it with images from the target class that drive it
        # strongly; a negative diff means it decreased away from the source,
        # illustrated the same way but from the source class. Matches the
        # convention already established in visualize_concepts.py.
        ref_class = target_idx if diff > 0 else source_idx
        refs = fv.get_stats_reference(
            int(concept), concept_layer, [ref_class], attr, (0, refs_per_concept),
            rf=True, composite=composite, plot_fn=vis_opaque_img,
        )
        # channel_grad's spatial dims match the *uncropped* canvas (gradient
        # was computed on the full letterboxed input) -- crop identically so
        # it stays pixel-aligned with orig_hwc/cf_hwc for the overlay.
        heat = localization_heatmap(orig_full, spatial_grad[int(concept)], diff, spatial_th)[rows, cols]
        heat_cf = localization_heatmap(cf_full, spatial_grad_cf[int(concept)], diff, spatial_th)[rows, cols]
        thumb_composite = pack_thumbnails(next(iter(refs.values())))
        ref_rows.append((int(concept), float(diff), thumb_composite, heat, heat_cf))

    ref_cols = max(2, refs_per_concept)
    ncols = ref_cols + 2  # +2 for the original/counterfactual localization columns on the right
    nrows = 1 + len(ref_rows)
    half = ref_cols // 2

    fig = plt.figure(figsize=(2.8 * ncols, 2.2 * (nrows - 1) + 5.5), constrained_layout=True)
    gs = fig.add_gridspec(nrows, ncols, height_ratios=[3.2] + [0.85] * len(ref_rows))

    # Original takes the left half of the reference-image columns, counterfactual
    # the right half -- spanning columns (not just one cell each) is what
    # actually makes them bigger, not just taller with extra padding around
    # a small image. The diff map spans both localization columns below it
    # (otherwise unused in this row) for a modest size bump.
    ax_orig = fig.add_subplot(gs[0, 0:half])
    ax_cf = fig.add_subplot(gs[0, half:ref_cols])
    ax_diff = fig.add_subplot(gs[0, ref_cols:ref_cols + 2])
    # One wide axis per row for the packed reference-thumbnail strip, not
    # ref_cols separate equal-width cells -- see pack_thumbnails() above.
    concept_ref_axs = [fig.add_subplot(gs[r, 0:ref_cols]) for r in range(1, nrows)]
    concept_orig_loc_axs = [fig.add_subplot(gs[r, ref_cols]) for r in range(1, nrows)]
    concept_cf_loc_axs = [fig.add_subplot(gs[r, ref_cols + 1]) for r in range(1, nrows)]

    for ax in [ax_orig, ax_cf, ax_diff] + concept_ref_axs + concept_orig_loc_axs + concept_cf_loc_axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Short titles when the prediction matches the class being shown (the
    # common case), longer ones spelling out the mismatch otherwise.
    ax_orig.imshow(orig_hwc.numpy())
    if data_dict["in_pred"] == source:
        title0 = f"{source}\n({data_dict['in_confid']:.0%} conf.)"
    else:
        title0 = f"{source}\npred: {data_dict['in_pred']} ({data_dict['in_confid']:.0%})"
    ax_orig.set_title(f"Original\n{title0}", fontsize=18)

    ax_cf.imshow(cf_hwc.numpy())
    if data_dict["out_pred"] == target:
        title1 = f"{target}\n({data_dict['out_tgt_confid']:.0%} conf.)"
    else:
        title1 = (
            f"target: {target}\npred: {data_dict['out_pred']} "
            f"({data_dict['out_confid']:.0%}), P(target)={data_dict['out_tgt_confid']:.0%}"
        )
    ax_cf.set_title(f"Counterfactual\n{title1}", fontsize=18)

    ax_diff.imshow(diff_map)
    ax_diff.set_title("Diff: 1-SSIM", fontsize=18)

    for r, (concept, diff, thumb_composite, heat, heat_cf) in enumerate(ref_rows):
        color = add_color if diff > 0 else remove_color
        sign = "+" if diff > 0 else "−"
        direction = f"→ {target}" if diff > 0 else f"→ {source}"
        concept_ref_axs[r].set_title(f"{sign} Concept {concept}   {direction}   Δ={diff:.2f}", loc="left",
                                      fontsize=18, color=color)
        # Packed strip, not one thumbnail per equal-width cell -- aspect=
        # 'auto' stretches it to fill the axis exactly (imshow's default
        # 'equal' aspect preserves proportions by shrinking the box instead,
        # which under constrained_layout leaves a gap between the strip and
        # the localization columns well past what the strip's own aspect
        # ratio would predict -- constrained_layout reserves extra room for
        # this row's own title text that a naive figsize/ncols estimate
        # doesn't account for, so trying to precompute and compensate for it
        # was chasing a moving target). A packed montage's exact proportions
        # aren't meaningful the way a real photo's are, so a bit of stretch
        # is a fine trade for a guaranteed, gap-free fit.
        concept_ref_axs[r].imshow(thumb_composite, aspect="auto")
        concept_orig_loc_axs[r].imshow(heat)
        concept_cf_loc_axs[r].imshow(heat_cf)
        if r == 0:
            concept_orig_loc_axs[r].set_title("Localization\n(original)", fontsize=18)
            concept_cf_loc_axs[r].set_title("Localization\n(counterfactual)", fontsize=18)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_example(cfg, classifier_model, device, fv, composite, concept_layer, i2h, data_dict, uidx, save_path):
    """Paper-style single example, read left-to-right:

        [ Original ]            [ Concept 1 refs + loc ]           [ Counterfactual ]
        [ LRP(source) ]   -->   [ Concept 2 refs + loc ]   -->     [ LRP(target) ]
                                [ Concept 3 refs + loc ]
                                [ Concept 4 refs + loc ]

    Left column: the original image with, below it (slightly smaller), its LRP
    attribution for the *original* class. Middle: the top-4 most-changed
    concepts, each shown as its reference-image montage with the concept's
    localization on the original image behind it. Right: the counterfactual
    with, below it, the LRP attribution for the *target* class. Arrows fan out
    from the original to the four concepts and converge from the concepts onto
    the counterfactual.
    """
    conditions = np.asarray(data_dict["conditions"])
    concept_diff = np.asarray(data_dict["concept_diff"])

    n_concepts = min(cfg.get("num_example_concepts", 5), len(conditions))
    order = np.argsort(-np.abs(concept_diff))[:n_concepts]
    conditions, concept_diff = conditions[order], concept_diff[order]

    source, target = data_dict["source"], data_dict["target"]
    name_to_idx = {v: k for k, v in i2h.items()}
    source_idx, target_idx = name_to_idx[source], name_to_idx[target]

    orig_full = data_dict["image"].clamp(0, 1).permute(1, 2, 0)
    cf_full = data_dict["gen_image"].clamp(0, 1).permute(1, 2, 0)
    rows, cols = crop_letterbox_bounds(orig_full)
    orig_hwc = orig_full[rows, cols].numpy()
    cf_hwc = cf_full[rows, cols].numpy()

    # LRP for the decision actually being explained on each side: source class
    # on the original, target class on the counterfactual.
    lrp_src = lrp_class_heatmap(fv, composite, data_dict["image"], source_idx, rows, cols, device)
    lrp_tgt = lrp_class_heatmap(fv, composite, data_dict["gen_image"], target_idx, rows, cols, device)

    spatial_grad = localize_concepts(
        cfg, classifier_model, device, data_dict["image"], target_idx, concept_layer, len(i2h)
    )
    spatial_th = cfg.get("localization_spatial_th", 0.4)
    add_color, remove_color = "darkgreen", "tab:red"

    refs_per_concept = cfg.get("num_example_refs", 8)
    panels = []
    for concept, diff in zip(conditions, concept_diff):
        ref_class = target_idx if diff > 0 else source_idx
        refs = fv.get_stats_reference(
            int(concept), concept_layer, [ref_class], "relevance", (0, refs_per_concept),
            rf=True, composite=composite, plot_fn=vis_opaque_img,
        )
        heat = localization_heatmap(orig_full, spatial_grad[int(concept)], diff, spatial_th)[rows, cols]
        thumb = pack_thumbnails(next(iter(refs.values())))
        panels.append((int(concept), float(diff), thumb, heat))

    n = len(panels)
    fig_w, fig_h = 20, 1.3 * n + 3.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _bare(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # --- left column: original + LRP(source) below (slightly smaller) ---
    ax_orig = fig.add_axes([0.02, 0.55, 0.20, 0.38])
    ax_orig.imshow(orig_hwc)
    ax_orig.set_title(f"$\\bf{{Original}}$\n{source}\n({data_dict['in_confid']:.0%} conf.)", fontsize=16)
    _bare(ax_orig)
    ax_lrp_s = fig.add_axes([0.04, 0.10, 0.16, 0.34])
    ax_lrp_s.imshow(lrp_src)
    ax_lrp_s.set_title("Explanation (Original)", fontsize=16)
    _bare(ax_lrp_s)

    # --- right column: counterfactual + LRP(target) below ---
    ax_cf = fig.add_axes([0.78, 0.55, 0.20, 0.38])
    ax_cf.imshow(cf_hwc)
    ax_cf.set_title(f"$\\bf{{Counterfactual}}$\n{target}\n({data_dict['out_tgt_confid']:.0%} conf.)", fontsize=16)
    _bare(ax_cf)
    ax_lrp_t = fig.add_axes([0.80, 0.10, 0.16, 0.34])
    ax_lrp_t.imshow(lrp_tgt)
    ax_lrp_t.set_title("Explanation (Counterfactual)", fontsize=16)
    _bare(ax_lrp_t)

    # --- middle: one row per concept -> [concept montage] [localization];
    # the montage box height is derived from the montage's own aspect ratio
    # so the reference thumbnails keep their true proportions (no vertical
    # squashing), centred within an evenly spaced row slot. ---
    mid_left, refs_w = 0.30, 0.36
    gap_rl = 0.015
    loc_left = mid_left + refs_w + gap_rl
    top, bottom = 0.86, 0.06
    slot = (top - bottom) / n

    # Spatial-conditioning box: sized to the heatmap's own aspect ratio
    # rather than a fixed-width box, so imshow's default 'equal' aspect
    # (which shrinks the image to fit *inside* a mismatched box, centred,
    # leaving whitespace on whichever side doesn't bind) has nothing to pad
    # -- the box is exactly the image's shape, just scaled up. Every row
    # shares the same crop (rows/cols is computed once from orig_full), so
    # one aspect ratio -- taken from the first panel -- fits them all. Sized
    # to fill most of the row slot vertically, then capped by whatever
    # horizontal room is actually left before the counterfactual column.
    a_h = panels[0][3].shape[1] / panels[0][3].shape[0]  # heatmap width/height
    avail_w = 0.78 - loc_left - 0.01
    heat_h = slot * 0.92
    heat_w = heat_h * fig_h * a_h / fig_w
    if heat_w > avail_w:
        heat_w = avail_w
        heat_h = heat_w * fig_w / a_h / fig_h

    centers = []
    for i, (concept, diff, thumb, heat) in enumerate(panels):
        a_m = thumb.shape[1] / thumb.shape[0]
        h_m = min((refs_w * fig_w / a_m) / fig_h, slot * 0.9)
        yc = top - (i + 0.5) * slot
        y0 = yc - h_m / 2
        centers.append(yc)
        color = add_color if diff > 0 else remove_color
        sign = "+" if diff > 0 else "\u2212"

        ax_r = fig.add_axes([mid_left, y0, refs_w, h_m])
        ax_r.imshow(thumb, aspect="auto")  # box matches montage aspect -> undistorted
        ax_r.set_title(f"{sign} Concept {concept}   \u0394={diff:.2f}", loc="left", fontsize=16, color=color)
        _bare(ax_r)
        ax_h = fig.add_axes([loc_left, yc - heat_h / 2, heat_w, heat_h])
        ax_h.imshow(heat)  # box matches the heatmap's aspect -> fills it exactly, no whitespace
        _bare(ax_h)

    # --- column headers ---
    infl_x = (0.22 + mid_left) / 2 + 0.05
    fig.text(infl_x, 0.93, "Positive / Negative\nConcept Influence",
             ha="center", va="center", fontsize=16)
    fig.text(mid_left + refs_w / 2, 0.93, "Concept Visualization",
             ha="center", va="center", fontsize=16)
    fig.text(loc_left + heat_w / 2, 0.93, "Spatial Conditioning",
             ha="center", va="center", fontsize=16)

    # --- arrows: original -> each concept, coloured green/red by the sign of
    # the concept influence; an S-curve (horizontal at both ends via a cubic
    # Bezier with control points sharing the endpoints' y) so the fan reads
    # cleanly ---
    ov = fig.add_axes([0, 0, 1, 1])
    ov.set_xlim(0, 1)
    ov.set_ylim(0, 1)
    ov.axis("off")
    ov.set_zorder(5)
    orig_right = (0.22, 0.72)
    concept_left = mid_left - 0.004
    for (concept, diff, _thumb, _heat), yc in zip(panels, centers):
        color = add_color if diff > 0 else remove_color
        x0, y0 = orig_right
        x1, y1 = concept_left, yc
        dx = x1 - x0
        path = Path(
            [(x0, y0), (x0 + 0.45 * dx, y0), (x1 - 0.45 * dx, y1), (x1, y1)],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
        )
        ov.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>",
                                     mutation_scale=22, color=color, lw=2.0, shrinkA=0, shrinkB=0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base = os.path.splitext(save_path)[0]
    for ext in ("png", "svg"):
        fig.savefig(f"{base}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {base}.png / {base}.svg")


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    classifier_model.train = disabled_train

    concept_layer = resolve_layer(classifier_model, cfg.concept_layer)
    fv, composite, ref_dataset = build_feature_visualization(cfg, classifier_model)
    i2h = get_class_names(cfg, ref_dataset)

    out_dir = out_dir_for(cfg)

    if "uidx" in cfg:
        uidx_list = list(cfg.uidx)
    else:
        available = sorted(
            int(os.path.splitext(f)[0])
            for f in os.listdir(out_dir)
            if f.endswith(".pth") and os.path.splitext(f)[0].isdigit()
        )
        uidx_list = available[: cfg.get("num_summary_samples", 10)]

    example = cfg.get("example", False)
    if example:
        save_dir = f"/results/counterfactuals/examples/{dataset_tag(cfg)}_{cfg.classifier_model.name}"
        plot_fn = plot_example
    else:
        save_dir = f"/results/counterfactuals/fv_images/summaries_{dataset_tag(cfg)}_{cfg.classifier_model.name}"
        plot_fn = plot_summary
    for uidx in uidx_list:
        dict_save_path = os.path.join(out_dir, f"{str(uidx).zfill(5)}.pth")
        if not os.path.isfile(dict_save_path):
            print(f"skipping {uidx}: {dict_save_path} not found")
            continue
        data_dict = torch.load(dict_save_path, map_location="cpu")
        if "conditions" not in data_dict:
            print(f"skipping {uidx}: no saved 'conditions' (was this generated with run_concept_ldce.py?)")
            continue
        save_path = os.path.join(save_dir, f"{str(uidx).zfill(5)}.png")
        plot_fn(cfg, classifier_model, device, fv, composite, concept_layer, i2h, data_dict, uidx, save_path)


if __name__ == "__main__":
    main()
