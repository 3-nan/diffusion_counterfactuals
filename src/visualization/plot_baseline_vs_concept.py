"""Side-by-side comparison grid: original image, baseline counterfactual and
concept-based (CoLa-DCE) counterfactual, for several examples from different
classes.

Both the baseline (run_ldce_baseline.py) and the concept run
(run_concept_ldce.py) save one .pth per sample under
    <output_dir>/bucket_<shard>_<num_shards>/<uidx>.pth
keyed by the *same* uidx (unique_data_idx * num_shards + shard), so a given
uidx is the same source sample in both -- that's what lets us line up the two
counterfactuals for one original side by side. Each dict carries "image"
(original), "gen_image" (counterfactual), "source"/"target" class names and
the in/out confidences; this script only reads those, no classifier or CRP
pass required.

By default it scans the shared uidx range and picks the first N samples with
*distinct source classes* (so the grid spans different classes) for which both
runs succeeded. Pass explicit ids with --uidx to override.

Usage (host or any container with torch/matplotlib):
    python src/visualization/plot_baseline_vs_concept.py \
        --baseline /results/counterfactuals/cars_vgg16bn_baseline/bucket_0_1 \
        --concept  /results/counterfactuals/cars_vgg16bn_concept/bucket_0_1 \
        --out      /results/counterfactuals/examples/compare_cars_vgg16_bn \
        --n 8
    python src/visualization/plot_baseline_vs_concept.py ... --uidx 0 3 5 9
"""
import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


def crop_letterbox_bounds(image_hwc, fill=114 / 255, tol=0.03):
    """Row/col bounds trimming letterbox_resize()'s flat-gray padding, computed
    from the original (guaranteed flat there) and reused for both
    counterfactuals since all three share the same pixel-aligned canvas."""
    is_pad = (image_hwc - fill).abs().amax(dim=-1) < tol
    rows = (~is_pad.all(dim=1)).nonzero(as_tuple=True)[0]
    cols = (~is_pad.all(dim=0)).nonzero(as_tuple=True)[0]
    if len(rows) == 0 or len(cols) == 0:
        return slice(None), slice(None)
    return slice(rows.min().item(), rows.max().item() + 1), slice(cols.min().item(), cols.max().item() + 1)


def load_sample(bucket_dir, uidx):
    path = os.path.join(bucket_dir, f"{str(uidx).zfill(5)}.pth")
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location="cpu")


def hwc(image_chw, rows, cols):
    return image_chw.clamp(0, 1).permute(1, 2, 0)[rows, cols].numpy()


def select_batches(baseline_dir, concept_dir, n, num_figures, scan_limit=1000):
    """Produce num_figures batches of n uidx each. Within a batch the source
    classes are distinct; candidates are visited in a strided order so the
    picks are spread across the whole dataset rather than clustered at the
    start, and the scan continues across batches so successive figures draw
    from different samples."""
    stride = max(1, scan_limit // (num_figures * n))
    order = []
    for off in range(stride):
        order.extend(range(off, scan_limit, stride))

    batches, pos = [], 0
    for _ in range(num_figures):
        chosen, seen = [], set()
        while pos < len(order) and len(chosen) < n:
            uidx = order[pos]
            pos += 1
            cpath = os.path.join(concept_dir, f"{str(uidx).zfill(5)}.pth")
            bpath = os.path.join(baseline_dir, f"{str(uidx).zfill(5)}.pth")
            if os.path.isfile(cpath) and os.path.isfile(bpath):
                src = torch.load(cpath, map_location="cpu")["source"]
                if src not in seen:
                    seen.add(src)
                    chosen.append(uidx)
        if chosen:
            batches.append(sorted(chosen))
    return batches


def draw_figure(baseline_dir, concept_dir, uidx_list, out):
    n = len(uidx_list)
    col_titles = ["Original", "Baseline", "CoLa-DCE"]
    fig, axs = plt.subplots(n, 3, figsize=(3 * 3.0, 3.0 * n))
    axs = np.atleast_2d(axs)
    # Fix the axes geometry up front so the per-row get_position() below (used
    # to centre the target label between the two counterfactual columns)
    # reflects the final layout.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.03, wspace=0.05, hspace=0.02)

    for r, uidx in enumerate(uidx_list):
        base = load_sample(baseline_dir, uidx)
        conc = load_sample(concept_dir, uidx)
        if base is None or conc is None:
            for c in range(3):
                axs[r, c].axis("off")
            continue

        orig_full = conc["image"].clamp(0, 1).permute(1, 2, 0)
        rows, cols = crop_letterbox_bounds(orig_full)
        orig = orig_full[rows, cols].numpy()
        base_cf = hwc(base["gen_image"], rows, cols)
        conc_cf = hwc(conc["gen_image"], rows, cols)

        source, target = conc["source"], conc["target"]
        for c, img in enumerate((orig, base_cf, conc_cf)):
            ax = axs[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=16, fontweight="bold")
        # source under the original, target once and centred between the two
        # counterfactual columns.
        axs[r, 0].set_xlabel(source, fontsize=12)
        p_base = axs[r, 1].get_position()
        p_conc = axs[r, 2].get_position()
        fig.text((p_base.x0 + p_conc.x1) / 2, p_base.y0 - 0.005, target,
                 ha="center", va="top", fontsize=12)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {out}.png / {out}.svg  (uidx={uidx_list})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="baseline bucket dir")
    ap.add_argument("--concept", required=True, help="concept bucket dir")
    ap.add_argument("--out", required=True, help="output path stem (no extension)")
    ap.add_argument("--n", type=int, default=8, help="examples per figure")
    ap.add_argument("--num-figures", type=int, default=6, help="how many figures to emit")
    ap.add_argument("--uidx", type=int, nargs="*", default=None,
                    help="explicit sample ids for a single figure (overrides --num-figures)")
    args = ap.parse_args()

    if args.uidx:
        draw_figure(args.baseline, args.concept, args.uidx, args.out)
        return

    batches = select_batches(args.baseline, args.concept, args.n, args.num_figures)
    if not batches:
        raise SystemExit("no overlapping samples found in the two bucket dirs")
    for i, uidx_list in enumerate(batches, start=1):
        draw_figure(args.baseline, args.concept, uidx_list, f"{args.out}_{i}")


if __name__ == "__main__":
    main()
