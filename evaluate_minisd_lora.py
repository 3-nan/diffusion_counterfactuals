"""Evaluation targets for ``finetune_minisd_lora.py``.

Run this twice -- once without ``--lora-path`` (stock miniSD) and once with
your trained adapter -- and compare the three numbers it prints. See the
"Evaluation targets" section of ``finetune_minisd_lora.py``'s docstring for
what each one means and what to expect; the short version:

1. Held-out denoising loss (always computed) -- cheap sanity check, should
   drop with the LoRA active.
2. FID against real BoxCars116k test images (always computed) -- the main
   target, measures whether generations now look like the BoxCars domain.
3. Classifier agreement (only if ``--classifier-path`` is given) -- whether
   the caption used to generate an image actually matches what a
   BoxCars-trained classifier (from ``finetune_classifier.py --dataset
   boxcars``) thinks it sees. Needs a classifier trained with the same
   ``--part``, since class indices must line up with ``BoxCars116k.classes``.

Usage
-----
    # stock miniSD baseline
    python evaluate_minisd_lora.py --root /data/boxcars --out-dir /results/eval/minisd_stock

    # trained LoRA
    python evaluate_minisd_lora.py --root /data/boxcars \\
        --lora-path /results/models/minisd_lora_boxcars/minisd_lora_boxcars_hard_final.pt \\
        --out-dir /results/eval/minisd_lora \\
        --classifier-path /results/models/vgg16bn_boxcars/<ckpt>.pth
"""
import argparse
import json
import os
import random
import shutil
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ldce.sampling_helpers import get_model, disabled_train, _unmap_img

from data.datasets import BoxCars116k, letterbox_resize
from finetune_minisd_lora import (
    CAPTION_TEMPLATE, inject_lora, load_lora_state_dict, sample_images,
)
from finetune_classifier import IMAGENET_MEAN, IMAGENET_STD


def make_eval_transform(image_size):
    # No augmentation, no flip -- deterministic so repeated runs (stock vs.
    # LoRA) see the exact same real images. letterbox, not CenterCrop -- a
    # third of BoxCars116k test crops are rectangular enough (mean long/short
    # ratio 1.26, up to 2.3) that CenterCrop trims real car content, which
    # would corrupt both the FID reference images and the classifier-
    # agreement check (finetune_classifier.py's val_tf uses the same
    # letterbox helper, so this keeps the classifier-agreement metric on the
    # distribution that classifier actually expects).
    return transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def held_out_denoising_loss(model, loader, device, seed, max_batches):
    losses = []
    for i, (images, labels) in enumerate(tqdm(loader, desc="denoising loss", total=min(max_batches, len(loader)))):
        if i >= max_batches:
            break
        # fixed per-batch seed -> same noise/timesteps for stock vs. LoRA runs,
        # so the comparison isn't confounded by sampling variance.
        torch.manual_seed(seed + i)
        images = images.to(device)
        captions = [CAPTION_TEMPLATE.format(loader.dataset.classes[l.item()]) for l in labels]

        z0 = model.get_first_stage_encoding(model.encode_first_stage(_unmap_img(images)))
        cond = model.get_learned_conditioning(captions)
        # torch.manual_seed(seed + i) above already seeds this draw.
        t = torch.randint(0, model.num_timesteps, (z0.shape[0],), device=device).long()
        noise = torch.randn(z0.shape, device=device)
        z_noisy = model.q_sample(x_start=z0, t=t, noise=noise)
        model_output = model.apply_model(z_noisy, t, cond)
        target = noise if model.parameterization == "eps" else z0
        losses.append(F.mse_loss(model_output, target).item())
    return sum(losses) / len(losses)


@torch.no_grad()
def fid_and_classifier_agreement(model, test_set, device, out_dir, num_images, batch_size,
                                 image_size, ddim_steps, scale, seed, classifier_path):
    real_dir = os.path.join(out_dir, "real")
    gen_dir = os.path.join(out_dir, "generated")
    for d in (real_dir, gen_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)

    random.seed(seed)
    indices = random.sample(range(len(test_set)), min(num_images, len(test_set)))

    classifier = None
    if classifier_path:
        classifier = torch.load(classifier_path, map_location=device, weights_only=False)
        classifier.to(device).eval()
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    n_correct, n_total = 0, 0
    n_written = 0
    for start in tqdm(range(0, len(indices), batch_size), desc="FID/classifier generation"):
        batch_idx = indices[start:start + batch_size]
        real_images, labels = zip(*(test_set[i] for i in batch_idx))
        labels = torch.tensor(labels)
        captions = [CAPTION_TEMPLATE.format(test_set.classes[l.item()]) for l in labels]

        gen = sample_images(model, captions, image_size=image_size, ddim_steps=ddim_steps,
                            scale=scale, seed=seed + start, device=device)

        for j in range(len(batch_idx)):
            save_image(real_images[j].clip(0, 1), os.path.join(real_dir, f"{n_written:06d}.png"))
            save_image(gen[j].clip(0, 1), os.path.join(gen_dir, f"{n_written:06d}.png"))
            n_written += 1

        if classifier is not None:
            logits = classifier(normalize(gen).to(device))
            pred = logits.argmax(dim=1).cpu()
            n_correct += (pred == labels).sum().item()
            n_total += len(labels)

    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths([real_dir, gen_dir], batch_size=50, device=device, dims=2048)

    agreement = n_correct / n_total if n_total > 0 else None
    return fid, agreement


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="BoxCars116k root")
    p.add_argument("--part", default="hard", choices=BoxCars116k.PARTS)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cfg-path", default="configs/stable-diffusion/v1-inference.yaml")
    p.add_argument("--ckpt-path", default="models/ldm/miniSD/miniSD.ckpt")
    p.add_argument("--lora-path", default=None, help="omit to evaluate the stock (non-finetuned) checkpoint")
    p.add_argument("--classifier-path", default=None,
                   help="BoxCars VGG16bn checkpoint from finetune_classifier.py; omit to skip metric 3")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--denoising-loss-batches", type=int, default=100)
    p.add_argument("--fid-num-images", type=int, default=500)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model = get_model(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path).to(device)
    model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad_(False)

    if args.lora_path:
        ckpt = torch.load(args.lora_path, map_location="cpu")
        inject_lora(model.model.diffusion_model, rank=ckpt["rank"], alpha=ckpt["alpha"])
        load_lora_state_dict(model.model.diffusion_model, ckpt["lora_state_dict"])
        print(f"evaluating LoRA checkpoint: {args.lora_path}")
    else:
        print("evaluating stock (non-finetuned) checkpoint")

    test_set = BoxCars116k(root=args.root, transform=make_eval_transform(args.image_size),
                           split="test", part=args.part, return_index=False)
    print(f"BoxCars116k ({args.part}) test: {len(test_set)} samples, {len(test_set.classes)} classes")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers)

    denoising_loss = held_out_denoising_loss(model, test_loader, device, args.seed,
                                             args.denoising_loss_batches)
    print(f"1. held-out denoising loss: {denoising_loss:.4f}")

    fid, agreement = fid_and_classifier_agreement(
        model, test_set, device, args.out_dir, args.fid_num_images, args.batch_size,
        args.image_size, args.ddim_steps, args.scale, args.seed, args.classifier_path,
    )
    print(f"2. FID vs. real BoxCars116k test images: {fid:.2f}")
    if agreement is not None:
        print(f"3. classifier top-1 agreement: {agreement:.3f}")
    else:
        print("3. classifier top-1 agreement: skipped (no --classifier-path)")

    report = {
        "lora_path": args.lora_path,
        "classifier_path": args.classifier_path,
        "held_out_denoising_loss": denoising_loss,
        "fid": fid,
        "classifier_agreement": agreement,
        "fid_num_images": args.fid_num_images,
        "denoising_loss_batches": args.denoising_loss_batches,
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'report.json')}")


if __name__ == "__main__":
    sys.exit(main())
