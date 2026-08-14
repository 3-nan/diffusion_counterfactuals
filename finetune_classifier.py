"""Finetune VGG16bn (or plain VGG16) on Stanford Cars -- adaptable to the other
fine-grained vehicle datasets in ``data/datasets.py`` -- and save a checkpoint
that ``compute_closest_indices.py`` / the LDCE configs can load directly.

Why VGG16bn, why ``features.37``
---------------------------------
VGG16bn is the right default for a boring but decisive reason: it's what the
CoLa-DCE paper used, and ``features.37`` -- the layer hardcoded in their
config and in ``compute_closest_indices.py`` -- is the second conv of
VGG16bn's last block. 512 channels at 14x14 (224 input) or 16x16 (256 input).
That's the sweet spot for CoLa-DCE: enough channels to give you distinct
nameable concepts, enough spatial resolution for the per-concept localization
masks to actually be informative. Plain sequential VGG also has no skip
connections, so relevance flow in CRP is clean.

Practical notes baked into the defaults below
----------------------------------------------
* Train at 256, not 224. Explanations run at the LDM's resolution -- a
  classifier trained at 224 and evaluated at 256 gives a slightly
  off-distribution gradient, which is exactly the signal steering the
  diffusion.
* Full finetune, not a linear probe. Stanford Cars has ~40 images/class; the
  conv features have to adapt or the concepts stay ImageNet-generic (you'll
  get "wheel" and "windshield", not "hexagonal grille mesh"). Every parameter
  is trainable here.
* SGD, lr=1e-3 for the pretrained body / 1e-2 for the freshly-initialised
  head, momentum 0.9, cosine schedule, 30 epochs, letterbox_resize(256) then
  RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(1, 1)) + hflip -- letterbox
  first so training sees the same padded, undistorted framing val/generation
  do, square-locked crop on top for mild augmentation without reintroducing
  distortion. Colour jitter is kept mild on purpose -- you're
  about to ask CRP which concepts separate two classes, and heavy jitter
  trains the model to ignore paint and trim colour, a real discriminative
  feature for vehicles.
* Don't chase accuracy. The paper's most interesting result (Sec. 5.4) is
  explaining misclassifications; a 97%-accurate model gives you almost
  nothing to explain. ~85-90% on Stanford Cars is both easy to hit with the
  above recipe and more useful -- this script trains for a fixed budget and
  reports where you landed rather than trying to squeeze out extra points.

Adding a dataset later
-----------------------
Register it in ``DATASET_BUILDERS`` below: a name -> function mapping that
takes ``(root, split, transform, **kwargs)`` and returns a ``Dataset`` with a
``.classes`` list. ``cars`` / ``compcars`` / ``boxcars`` already do this via
the classes in ``data/datasets.py``; ``imagefolder`` is a generic fallback
for anything laid out as ``<root>/<split>/<class>/<img>``.

Example
-------
    python finetune_classifier.py \
        --dataset cars --root /data/stanford_cars \
        --out-dir /results/models/vgg16bn_cars
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.vgg import vgg16, vgg16_bn
from torchvision.models.resnet import resnet18
from torchvision.models.vision_transformer import vit_b_16
from tqdm import tqdm

from data.datasets import letterbox_resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ARCHS = {"vgg16_bn": vgg16_bn, "vgg16": vgg16, "resnet18": resnet18, "vit_b_16": vit_b_16}


# ---------------------------------------------------------------------------
# Dataset registry -- the "adaptable to other datasets" part. Each builder
# takes (root, split, transform, **kwargs) and returns a Dataset exposing
# .classes (list[str]) and standard (img, label[, index]) items.
# ---------------------------------------------------------------------------

def _build_cars(root, split, transform, **kw):
    from data.datasets import StanfordCars
    return StanfordCars(root=root, transform=transform, split=split, return_index=False, **kw)


def _build_compcars(root, split, transform, **kw):
    from data.datasets import CompCars
    return CompCars(root=root, transform=transform, split=split, return_index=False, **kw)


def _build_boxcars(root, split, transform, **kw):
    from data.datasets import BoxCars116k
    return BoxCars116k(root=root, transform=transform, split=split, return_index=False, **kw)


def _build_imagefolder(root, split, transform, **kw):
    from torchvision import datasets as tv_datasets
    _ = kw
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir) and split == "test":
        # a lot of ImageFolder-style dumps call the held-out split "val"
        split_dir = os.path.join(root, "val")
    return tv_datasets.ImageFolder(split_dir, transform=transform)


DATASET_BUILDERS = {
    "cars": _build_cars,
    "compcars": _build_compcars,
    "boxcars": _build_boxcars,
    "imagefolder": _build_imagefolder,
}


def build_dataset(name, root, split, transform, **kw):
    if name not in DATASET_BUILDERS:
        raise ValueError(f"Unknown --dataset '{name}'. Known: {list(DATASET_BUILDERS)}")
    return DATASET_BUILDERS[name](root, split, transform, **kw)


def make_transforms(image_size):
    # letterbox FIRST, then crop-augment -- train_tf used to go straight to
    # RandomResizedCrop(scale=(0.7, 1.0)), which crops a sub-region and
    # stretches it to fill the whole frame and so never produces the gray
    # padding bars letterbox_resize introduces for non-square crops. That
    # meant training never showed the model a single padded image while
    # val_tf (below) and every downstream steering/generation call were 100%
    # padded -- a real train/val domain gap, not overfitting, even though it
    # looks like classic overfitting in the loss curves (train acc 99.7%, val
    # acc collapsed to 58.5% after this file's val_tf switched to letterbox).
    # RandomResizedCrop still runs afterward for mild scale augmentation, but
    # scale is tightened to (0.85, 1.0) and ratio pinned to square -- cropping
    # more aggressively risks cropping straight through the car on one axis
    # while only trimming padding on the other (the letterbox already "spent"
    # some of the frame on padding, unlike the old CenterCrop pipeline), and
    # unpinned ratio would reintroduce the exact distortion letterbox exists
    # to avoid.
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        # Mild on purpose -- paint/trim colour is a real discriminative
        # feature for vehicles and CRP will later be asked to point at it.
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    # letterbox (resize-to-fit + pad), not Resize+CenterCrop -- CenterCrop
    # trims whatever sticks out past the shorter side, which for a landscape
    # crop_to_bbox car crop can cut the nose/tail off. run_ldce_baseline.py
    # and compute_closest_indices.py use the same helper so the classifier is
    # trained, indexed and steered on identical preprocessing.
    val_tf = transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def head_module(model):
    """Return the final classification Linear layer for the supported archs.
    VGG exposes it as ``classifier[6]``; ResNet as ``fc``; torchvision ViT as
    ``heads.head``."""
    if hasattr(model, "fc"):         # ResNet family
        return model.fc
    if hasattr(model, "heads"):      # torchvision ViT (vit_b_16)
        return model.heads.head
    return model.classifier[6]        # VGG family


def build_model(arch, num_classes):
    if arch not in ARCHS:
        raise ValueError(f"Unknown --arch '{arch}'. Known: {list(ARCHS)}")
    model = ARCHS[arch](pretrained=True)
    # Swap in a freshly initialised head sized to this dataset, wherever the
    # arch keeps it (ResNet: model.fc, ViT: model.heads.head, VGG:
    # model.classifier[6]).
    if hasattr(model, "fc"):         # ResNet family
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "heads"):    # torchvision ViT (vit_b_16)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:                            # VGG family
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model


def param_groups(model, lr, head_lr):
    """Everything trains (full finetune, not a linear probe); the freshly
    initialised final layer just gets a higher learning rate than the
    pretrained body so it can catch up."""
    head_params = list(head_module(model).parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": body_params, "lr": lr},
        {"params": head_params, "lr": head_lr},
    ]


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, desc="train" if train else "val", leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_BUILDERS))
    p.add_argument("--root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arch", default="vgg16_bn", choices=list(ARCHS))
    p.add_argument("--image-size", type=int, default=None,
                   help="train/eval resolution; default 256 for conv nets, "
                        "forced to 224 for vit_b_16 (fixed patch grid)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--optimizer", default=None, choices=["sgd", "adamw"],
                   help="default sgd for conv nets, adamw for vit_b_16")
    p.add_argument("--lr", type=float, default=None,
                   help="body lr (default: sgd 1e-3, adamw 1e-4)")
    p.add_argument("--head-lr", type=float, default=None,
                   help="fresh head lr (default: sgd 1e-2, adamw 1e-3)")
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="default: sgd 1e-4, adamw 0.05")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    # passed straight through to the dataset builder (e.g. --min-box-size,
    # --bbox-padding for the vehicle datasets, --compcars-subset, etc.)
    p.add_argument("--dataset-kwargs", type=json.loads, default="{}",
                   help='JSON dict forwarded to the dataset builder, e.g. \'{"min_box_size": 64}\'')
    args = p.parse_args()

    # Arch-aware defaults. vit_b_16 has a fixed 224 patch grid (feeding 256 px
    # trips torchvision's positional-embedding assert) and trains far better
    # with AdamW + a lower lr than the SGD recipe tuned for the conv nets.
    is_vit = args.arch == "vit_b_16"
    if args.image_size is None:
        args.image_size = 224 if is_vit else 256
    if is_vit and args.image_size != 224:
        print(f"note: vit_b_16 requires 224 px input; overriding "
              f"--image-size {args.image_size} -> 224")
        args.image_size = 224
    if args.optimizer is None:
        args.optimizer = "adamw" if is_vit else "sgd"
    if args.lr is None:
        args.lr = 1e-4 if args.optimizer == "adamw" else 1e-3
    if args.head_lr is None:
        args.head_lr = 1e-3 if args.optimizer == "adamw" else 1e-2
    if args.weight_decay is None:
        args.weight_decay = 0.05 if args.optimizer == "adamw" else 1e-4

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf, val_tf = make_transforms(args.image_size)
    train_set = build_dataset(args.dataset, args.root, "train", train_tf, **args.dataset_kwargs)
    val_set = build_dataset(args.dataset, args.root, "test", val_tf, **args.dataset_kwargs)
    num_classes = len(train_set.classes)
    print(f"{args.dataset}: {len(train_set)} train / {len(val_set)} val samples, {num_classes} classes")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = build_model(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups(model, args.lr, args.head_lr),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            param_groups(model, args.lr, args.head_lr),
            momentum=args.momentum, weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history = []
    best_acc, best_state = 0.0, None

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"train loss {train_loss:.3f} acc {train_acc:.3f}  |  "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    ckpt_path = os.path.join(
        args.out_dir, f"{args.arch}_{args.dataset}_{timestamp}_{best_acc:.3f}.pth"
    )
    # Save the full module (not a state_dict) -- this is what
    # compute_closest_indices.py and the LDCE configs' classifier_path expect.
    torch.save(model, ckpt_path)
    with open(os.path.join(args.out_dir, f"{args.arch}_{args.dataset}_{timestamp}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val acc {best_acc:.3f} -> {ckpt_path}")

    if best_acc > 0.95:
        print("note: this classifier is very accurate. CoLa-DCE Sec. 5.4's most interesting "
              "results come from explaining misclassifications -- a near-perfect model gives "
              "you almost nothing to explain. Consider fewer epochs or a stronger crop scale "
              "range if you want more errors to work with; ~85-90% is the useful sweet spot.")


if __name__ == "__main__":
    sys.exit(main())
