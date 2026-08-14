"""Figure: counterfactual realism (FID) and concept-layer displacement.

Left panel shows FID (real vs. counterfactual images, from the results table);
right panel shows the scale-free concept-layer *cosine* distance read from the
``embed_distance.json`` files written by ``src.evaluation.analyze_results embed``.
Both are rendered as a paired baseline-vs-concept dot-and-CI plot per
architecture x dataset.

Usage:
    python -m src.visualization.plot_embedding_distance \
        --eval-root /results/eval --out /results/eval/embedding_distance
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (row label, architecture dir-tag, dataset dir-tag)
ARCHES = [("VGG-16-BN", "vgg16bn"), ("ResNet-18", "resnet18"), ("ViT-B/16", "vit_b_16")]
DATASETS = [("Stanford Cars", "cars"), ("BoxCars116k", "boxcars")]

BASELINE_COLOR = "#4C72B0"
CONCEPT_COLOR = "#C44E52"

# FID (value, ci_lo, ci_hi), n=1000, keyed by (arch_tag, dataset_tag, setting).
# Taken from the cross-architecture results table.
FID = {
    ("vgg16bn", "cars", "baseline"): (18.61, 17.45, 19.77),
    ("vgg16bn", "cars", "concept"): (18.69, 17.59, 19.80),
    ("vgg16bn", "boxcars", "baseline"): (22.03, 20.75, 23.32),
    ("vgg16bn", "boxcars", "concept"): (35.56, 34.17, 36.94),
    ("resnet18", "cars", "baseline"): (18.49, 17.34, 19.63),
    ("resnet18", "cars", "concept"): (18.56, 17.42, 19.70),
    ("resnet18", "boxcars", "baseline"): (22.44, 21.11, 23.78),
    ("resnet18", "boxcars", "concept"): (39.25, 37.62, 40.89),
    ("vit_b_16", "cars", "baseline"): (19.71, 18.45, 20.98),
    ("vit_b_16", "cars", "concept"): (19.93, 18.66, 21.20),
    ("vit_b_16", "boxcars", "baseline"): (23.38, 21.98, 24.77),
    ("vit_b_16", "boxcars", "concept"): (36.17, 34.74, 37.60),
}


def _load(eval_root, dataset_tag, arch_tag, setting):
    path = os.path.join(eval_root, f"{dataset_tag}_{arch_tag}_{setting}", "embed_distance.json")
    with open(path) as fh:
        return json.load(fh)


def _cos(entry, site):
    d = entry[site]
    return d["cosine"], d["cosine_ci_lo"], d["cosine_ci_hi"]


def build_rows(eval_root):
    """Ordered rows (top -> bottom): FID (left panel) + concept-layer cosine (right)."""
    rows = []
    for arch_name, arch_tag in ARCHES:
        for ds_name, ds_tag in DATASETS:
            base = _load(eval_root, ds_tag, arch_tag, "baseline")
            conc = _load(eval_root, ds_tag, arch_tag, "concept")
            rows.append({
                "label": f"{arch_name}\n{ds_name}",
                "fid": {"baseline": FID[(arch_tag, ds_tag, "baseline")],
                        "concept": FID[(arch_tag, ds_tag, "concept")]},
                "concept": {"baseline": _cos(base, "concept"),
                            "concept": _cos(conc, "concept")},
            })
    return rows


def _panel(ax, rows, site, title, xlabel):
    n = len(rows)
    y = list(range(n))[::-1]  # first row at top
    off = 0.16
    for yi, row in zip(y, rows):
        for setting, color, dy in (("baseline", BASELINE_COLOR, off),
                                   ("concept", CONCEPT_COLOR, -off)):
            val, lo, hi = row[site][setting]
            ax.errorbar(val, yi + dy, xerr=[[val - lo], [hi - val]],
                        fmt="o", ms=6, color=color, ecolor=color,
                        elinewidth=1.6, capsize=3, capthick=1.6, zorder=3)
    for yi in y:
        ax.axhline(yi, color="0.9", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.margins(x=0.12)
    ax.grid(axis="x", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="/results/eval")
    ap.add_argument("--out", default="/results/eval/embedding_distance")
    args = ap.parse_args()

    rows = build_rows(args.eval_root)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    _panel(axes[0], rows, "fid", "FID",
           "FID  (real vs. counterfactual images)")
    _panel(axes[1], rows, "concept", "Concept layer",
           "cosine distance  (1 - cos)  original vs. counterfactual")

    handles = [Line2D([0], [0], marker="o", ls="none", ms=7, color=BASELINE_COLOR, label="Baseline"),
               Line2D([0], [0], marker="o", ls="none", ms=7, color=CONCEPT_COLOR, label="CoLa-DCE")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Counterfactual realism (FID) and concept-layer displacement "
                 "(n=1000, 95% CI)", fontsize=12, y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    for ext in ("pdf", "png", "svg"):
        p = f"{args.out}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)


if __name__ == "__main__":
    main()
