"""Near-miss counterfactual target selection -> ``<name>_closest_indices.json``.

Implements the local target selection of CoLa-DCE (Sec. 4.1):

    y_c = f( argmin_{x' in X'} d(kappa(x'), kappa(x_hat))  s.t.  f(x') != f(x_hat) )

i.e. for each sample, encode it with the classifier's feature space kappa,
find the nearest reference sample the model puts in a *different* class, and
take that class as the counterfactual target. Output matches the format of
``pets_closest_indices.json`` / ``flowers_closest_indices.json``:

    {"<sample_idx>": [<target_cls>, <2nd>, <3rd>, <4th>, <5th>], ...}

keyed by dataset index, with the top-k distinct near-miss classes ranked by
distance. Keeping k>1 lets you fall back when the closest target refuses to
flip within the diffusion budget.

Example
-------
    # per-setting output: writes data/cars_closest_indices_resnet18.json
    python compute_closest_indices.py \
        --dataset cars --root /data/stanford_cars \
        --checkpoint /results/models/resnet18_cars/resnet18_cars_...pth \
        --name resnet18 --layer layer4.1.conv1 --top-k 5

    # legacy / VGG output (no --name): writes data/cars_closest_indices.json
    python compute_closest_indices.py \
        --dataset cars --root /data/stanford_cars \
        --checkpoint ./ckpt/vgg16bn_cars.pth \
        --layer features.37 --top-k 5

``--name`` should match ``classifier_model.name`` in the LDCE config so the run
scripts (via ``data.datasets.closest_indices_path``) pick the file up
automatically; without ``--name`` the shared legacy filename is written, which
is what the VGG runs already consume, so nothing about the VGG path changes.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


@torch.no_grad()
def encode(model, layer_name, loader, device):
    """Return (features [N, D], predictions [N], indices [N]).

    ``kappa`` is the globally average-pooled activation of ``layer_name``, which
    is the 'Act' variant in Table 1. For the stronger 'Attr' variant, swap this
    hook for an LRP/CRP backward pass and pool the relevances instead -- the
    rest of the script is unchanged.
    """
    feats = {}
    module = dict(model.named_modules())[layer_name]
    handle = module.register_forward_hook(
        lambda m, inp, out: feats.__setitem__("a", out)
    )

    all_f, all_p, all_i = [], [], []
    for batch in loader:
        img, label, idx = batch[0], batch[1], batch[2]
        logits = model(img.to(device))
        a = feats["a"]
        if a.dim() == 4:            # conv: [B, C, H, W] -> [B, C]
            a = a.mean(dim=(2, 3))
        elif a.dim() == 3:          # ViT tokens: [B, T, C] -> [B, C]
            a = a.mean(dim=1)
        all_f.append(a.flatten(1).cpu())
        all_p.append(logits.argmax(1).cpu())
        all_i.append(idx.cpu() if torch.is_tensor(idx) else torch.tensor(idx))
        _ = label

    handle.remove()
    return torch.cat(all_f), torch.cat(all_p), torch.cat(all_i)


def near_miss(feats, preds, top_k=5, metric="cosine", chunk=512):
    """For each sample, the top-k distinct classes of its nearest differing-class
    neighbours, ranked by ascending distance."""
    if metric == "cosine":
        feats = F.normalize(feats, dim=1)

    n = feats.shape[0]
    out = {}
    for start in range(0, n, chunk):
        block = feats[start:start + chunk]
        if metric == "cosine":
            dist = 1.0 - block @ feats.T
        else:
            dist = torch.cdist(block, feats)

        order = dist.argsort(dim=1)
        for row, global_i in enumerate(range(start, min(start + chunk, n))):
            own = preds[global_i].item()
            ranked, seen = [], set()
            for j in order[row].tolist():
                if j == global_i:
                    continue
                cls = preds[j].item()
                if cls == own or cls in seen:
                    continue
                seen.add(cls)
                ranked.append(cls)
                if len(ranked) == top_k:
                    break
            out[global_i] = ranked
    return out


def build_dataset(name, root, image_size, **kw):
    from data.datasets import StanfordCars, CompCars, BoxCars116k, letterbox_resize

    # letterbox (resize-to-fit + pad), not Resize+CenterCrop -- must match
    # finetune_classifier.py's val_tf and run_ldce_baseline.py's generation
    # transform exactly, since kappa (the feature space this near-miss search
    # runs in) has to be computed on the same preprocessing the classifier
    # will actually see at generation time.
    tf = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if name == "cars":
        return StanfordCars(root=root, transform=tf, split="test", **kw)
    if name == "compcars":
        return CompCars(root=root, transform=tf, split="test", **kw)
    if name == "boxcars":
        return BoxCars116k(root=root, transform=tf, split="test", **kw)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["cars", "compcars", "boxcars"])
    p.add_argument("--root", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="fine-tuned classifier (the model you are explaining)")
    p.add_argument("--layer", default="features.37",
                   help="feature layer for kappa; features.37 for VGG16bn")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--name", default=None,
                   help="classifier/setting name (e.g. resnet18, vit_b_16); must "
                        "match classifier_model.name in the LDCE config. When "
                        "given, the output is namespaced as "
                        "data/<dataset>_closest_indices_<name>.json so the run "
                        "scripts pick it up automatically. Omit for the legacy "
                        "shared file used by the VGG runs.")
    p.add_argument("--out", default=None,
                   help="explicit output path; overrides the --name-derived "
                        "default. Defaults to "
                        "data/<dataset>_closest_indices[_<name>].json")
    args = p.parse_args()

    if args.out is not None:
        out_path = args.out
    elif args.name:
        out_path = os.path.join("data", f"{args.dataset}_closest_indices_{args.name}.json")
    else:
        out_path = os.path.join("data", f"{args.dataset}_closest_indices.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(model, dict):
        raise SystemExit(
            "--checkpoint is a state_dict; load it into your architecture first "
            "and save the full module, or adapt this line."
        )
    model.eval().to(device)

    ds = build_dataset(args.dataset, args.root, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print(f"{args.dataset}: {len(ds)} samples, {len(ds.classes)} classes")

    feats, preds, _ = encode(model, args.layer, loader, device)
    print(f"kappa: {tuple(feats.shape)} from '{args.layer}'")

    targets = near_miss(feats, preds, top_k=args.top_k, metric=args.metric)
    short = sum(1 for v in targets.values() if len(v) < args.top_k)
    if short:
        print(f"warning: {short} samples got fewer than {args.top_k} targets "
              f"(too few distinct predicted classes)")

    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(targets.items())}, f, indent=4)
    print(f"wrote {len(targets)} entries -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
