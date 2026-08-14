"""LoRA-finetune miniSD (the 256px SD1.5-family checkpoint already used by
run_ldce_baseline.py) on BoxCars116k, conditioned on class-name captions.

Why LoRA over a from-scratch class-conditional LDM
-----------------------------------------------------
Stock miniSD has never seen a traffic-surveillance camera photo -- fisheye-ish
wide shots, harsh contrast, motion blur, licence-plate-adjacent framing. That
domain gap is exactly what makes the *baseline* (no finetune) counterfactuals
on this data look wrong even before any classifier guidance gets involved: the
diffusion prior keeps pulling generations back towards "generic photo of a
car", not "car as seen by a BoxCars-style roadside camera". A LoRA adapter is
the cheap fix for that -- a few low-rank matrices bolted onto the UNet's
attention projections, hours on one GPU, easy to swap in/out at generation
time. Training a class-conditional LDM from scratch would learn the domain
*and* the class semantics more thoroughly, but it needs real per-class sample
counts to not collapse -- BoxCars116k's 51.7k train images across 107 "hard"
classes (make+model+submodel+year) clears that bar the way Stanford Cars'
~4k/196 classes doesn't. Still overkill for what's actually needed here: the
diffusion model's job in CoLa-DCE/LDCE is to stay near the data manifold while
classifier gradients (from the *separately* trained VGG16bn) do the actual
class steering, so it only has to know "this looks like a BoxCars photo" and
"this looks like a {caption}" well enough to be steerable -- not to be a
standalone accurate classifier itself. LoRA on the cross- and self-attention
projections gets both of those (domain look-and-feel from self-attention,
caption-following from cross-attention) without touching the frozen base
weights other datasets' configs still expect.

What gets adapted
-------------------
Every ``CrossAttention`` block's ``to_q`` / ``to_k`` / ``to_v`` / ``to_out[0]``
projections in the UNet (``model.model.diffusion_model``) -- both self-
attention (attn1, texture/domain) and cross-attention (attn2, text-following)
sub-blocks share the same ``CrossAttention`` class in this codebase, so
``inject_lora`` patches both without needing to special-case them. Everything
else (the VAE, the CLIP text encoder, the rest of the UNet) stays frozen and
shared with the un-finetuned checkpoint other configs (``v1_cars.yaml``, ...)
load directly.

Captions are ``"a photo of a {class}."`` using the exact class strings
``data.datasets.BoxCars116k.classes`` returns for the ``part="hard"`` split
(e.g. ``"skoda octavia sedan mk3"``) -- no manual label list to maintain.
10% of captions get blanked to the empty string during training
(``--uncond-prob``) so the LoRA'd UNet keeps a usable unconditional branch;
without that, classifier-free guidance at generation time is sampling a
branch the adapter never actually saw during training.

Evaluation targets
--------------------
Generative finetuning has no single number as clean as classifier val
accuracy, so this pairs with ``evaluate_minisd_lora.py``, which reports three
things -- run it once against the stock checkpoint (omit ``--lora-path``) and
once against your trained adapter to see the deltas:

1. **Held-out denoising loss** -- mean epsilon-MSE over the BoxCars116k test
   split, same objective as training. Cheap, no sampling required. Should
   drop noticeably vs. the stock checkpoint; if it doesn't, the LoRA isn't
   learning the domain at all (check ``--lr`` / rank first).
2. **FID against real BoxCars test images** -- generate txt2img samples from
   held-out test-split captions, compare against real test images with
   ``pytorch-fid``. This is the actual target: it measures whether generated
   images now look like BoxCars-domain photos rather than generic web photos
   of cars. Expect a large drop vs. stock miniSD (which is badly
   out-of-domain here) and for it to keep improving with more steps/rank
   before it plateaus.
3. **Classifier agreement (optional, needs a trained BoxCars VGG16bn
   checkpoint from finetune_classifier.py --dataset boxcars)** -- run
   generated images back through that classifier and check whether its top-1
   prediction matches the class the caption asked for. This is the metric
   that actually matters for the downstream use case: CoLa-DCE/LDCE needs the
   *caption* to correlate with class identity, not just domain style, or the
   text conditioning isn't doing anything useful for the classifier-guided
   sampling later. A LoRA that nails FID but scores near chance here is
   learning "BoxCars-style photo" without learning "this specific make/model"
   -- still useful as a domain prior, but don't expect it to sharpen
   class-conditional generation on its own.

None of these are pass/fail gates -- they're the numbers to compare before
sinking more GPU time into rank/step sweeps.

Usage
-----
    python finetune_minisd_lora.py \\
        --root /data/boxcars --out-dir /results/models/minisd_lora_boxcars

Resuming / continuing training:
    python finetune_minisd_lora.py --root /data/boxcars \\
        --out-dir /results/models/minisd_lora_boxcars --resume /results/models/minisd_lora_boxcars/<ckpt>.pt
"""
import argparse
import copy
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from ldce.sampling_helpers import get_model, disabled_train, _map_img, _unmap_img
from ldm.models.diffusion.ddim import DDIMSampler

from data.datasets import BoxCars116k, letterbox_resize

CAPTION_TEMPLATE = "a photo of a {}."


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable rank-r bypass:
    ``base(x) + scale * up(down(x))``. ``lora_up`` is zero-initialised so the
    wrapped layer is numerically identical to the original at step 0."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_up(self.dropout(self.lora_down(x)))


LORA_QKV = ("to_q", "to_k", "to_v")


def inject_lora(unet: nn.Module, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
    """Wrap every ``CrossAttention``'s q/k/v/out projections in ``unet`` with
    LoRA (both self- and cross-attention blocks -- see module docstring).
    Returns the list of newly-created trainable parameters."""
    lora_params = []
    for module in unet.modules():
        if type(module).__name__ != "CrossAttention":
            continue
        for name in LORA_QKV:
            base = getattr(module, name)
            # nn.Linear defaults to CPU/fp32 -- match the frozen base layer's
            # device/dtype or the forward pass mixes devices the moment this
            # runs on an already-.to(device)'d model.
            wrapped = LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout).to(
                device=base.weight.device, dtype=base.weight.dtype)
            setattr(module, name, wrapped)
            lora_params += list(wrapped.lora_down.parameters()) + list(wrapped.lora_up.parameters())
        base_out = module.to_out[0]
        wrapped_out = LoRALinear(base_out, rank=rank, alpha=alpha, dropout=dropout).to(
            device=base_out.weight.device, dtype=base_out.weight.dtype)
        module.to_out[0] = wrapped_out
        lora_params += list(wrapped_out.lora_down.parameters()) + list(wrapped_out.lora_up.parameters())
    if not lora_params:
        raise RuntimeError("inject_lora found no CrossAttention modules -- wrong model object?")
    return lora_params


def lora_state_dict(unet: nn.Module) -> dict:
    """Extract only the trainable LoRA weights, keyed by dotted module path."""
    sd = {}
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu().clone()
            sd[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu().clone()
    return sd


def load_lora_state_dict(unet: nn.Module, sd: dict):
    """Load weights saved by ``lora_state_dict`` into an already-``inject_lora``'d UNet."""
    loaded = 0
    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            module.lora_down.weight.data.copy_(sd[f"{name}.lora_down.weight"])
            module.lora_up.weight.data.copy_(sd[f"{name}.lora_up.weight"])
            loaded += 1
    print(f"loaded LoRA weights into {loaded} modules")


# ---------------------------------------------------------------------------
# Sampling (shared with evaluate_minisd_lora.py) -- plain txt2img DDIM, not
# the img2img counterfactual-editing path (CCMDDIMSampler in run_ldce_baseline.py).
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_images(model, prompts, image_size=256, ddim_steps=50, scale=5.0, seed=None, device="cuda"):
    if seed is not None:
        torch.manual_seed(seed)
    sampler = DDIMSampler(model)
    bs = len(prompts)
    uncond = model.get_learned_conditioning(bs * [""])
    cond = model.get_learned_conditioning(prompts)
    f = 8  # KL-VAE downsampling factor (ddconfig ch_mult 1,2,4,4)
    shape = (model.channels, image_size // f, image_size // f)
    samples, _ = sampler.sample(
        S=ddim_steps, batch_size=bs, shape=shape, conditioning=cond,
        unconditional_guidance_scale=scale, unconditional_conditioning=uncond,
        eta=0., verbose=False,
    )
    images = model.decode_first_stage(samples)
    return _map_img(images).clamp(0, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_transform(image_size):
    # Was Resize(image_size) + RandomCrop(pad_if_needed=True) -- plain
    # shorter-side resize + crop, which (unlike letterbox) trims content off
    # BoxCars116k's more rectangular crops (up to ~2.3:1, see
    # evaluate_minisd_lora.py's make_eval_transform comment). That meant the
    # LoRA was trained on different car framing than what get_dataset()'s
    # BoxCars116k branch in run_ldce_baseline.py / run_concept_ldce.py and
    # finetune_classifier.py's train_tf actually feed at generation/inference
    # time (letterbox_resize -- full crop content, no stretch, pad instead of
    # crop). Matched here so train-time and inference-time geometry agree;
    # RandomResizedCrop mirrors finetune_classifier.py's train_tf mild scale
    # augmentation (no ColorJitter/Normalize -- those are classifier-specific,
    # this feeds the VAE, not an ImageNet-normalized classifier).
    return transforms.Compose([
        transforms.Lambda(lambda img: letterbox_resize(img, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def training_step(model, images, captions, uncond_prob, device):
    images = images.to(device)
    with torch.no_grad():
        z0 = model.get_first_stage_encoding(model.encode_first_stage(_unmap_img(images))).detach()

    captions = ["" if torch.rand(()).item() < uncond_prob else c for c in captions]
    cond = model.get_learned_conditioning(captions)

    t = torch.randint(0, model.num_timesteps, (z0.shape[0],), device=device).long()
    noise = torch.randn_like(z0)
    z_noisy = model.q_sample(x_start=z0, t=t, noise=noise)
    model_output = model.apply_model(z_noisy, t, cond)

    target = noise if model.parameterization == "eps" else z0
    return F.mse_loss(model_output, target)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="BoxCars116k root (dataset.pkl, images/, ...)")
    p.add_argument("--part", default="hard", choices=BoxCars116k.PARTS)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cfg-path", default="configs/stable-diffusion/v1-inference.yaml")
    p.add_argument("--ckpt-path", default="models/ldm/miniSD/miniSD.ckpt")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--uncond-prob", type=float, default=0.1,
                   help="fraction of captions blanked for classifier-free-guidance training")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500, help="steps between checkpoint + preview saves")
    p.add_argument("--preview-classes", type=int, default=4)
    p.add_argument("--ddim-steps", type=int, default=50, help="only for preview sampling, not training")
    p.add_argument("--scale", type=float, default=5.0, help="cfg scale for preview sampling")
    p.add_argument("--resume", default=None, help="path to a checkpoint .pt from this script to resume from")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading base model: {args.cfg_path} / {args.ckpt_path}")
    model = get_model(cfg_path=args.cfg_path, ckpt_path=args.ckpt_path).to(device)
    model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad_(False)

    lora_params = inject_lora(model.model.diffusion_model, rank=args.rank, alpha=args.alpha,
                              dropout=args.lora_dropout)
    n_trainable = sum(p.numel() for p in lora_params)
    print(f"LoRA rank={args.rank} alpha={args.alpha}: {n_trainable / 1e6:.2f}M trainable params "
         f"across {len(lora_params) // 2} projections")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        load_lora_state_dict(model.model.diffusion_model, ckpt["lora_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"resumed from {args.resume} at step {start_step}")

    train_set = BoxCars116k(root=args.root, transform=make_transform(args.image_size),
                            split="train", part=args.part, return_index=False)
    print(f"BoxCars116k ({args.part}): {len(train_set)} train samples, {len(train_set.classes)} classes")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):
        if step >= args.warmup_steps:
            return args.lr
        return args.lr * (step + 1) / max(1, args.warmup_steps)

    preview_classes = list(range(0, len(train_set.classes), max(1, len(train_set.classes) // args.preview_classes)))[:args.preview_classes]
    preview_prompts = [CAPTION_TEMPLATE.format(train_set.classes[c]) for c in preview_classes]
    print(f"preview classes: {preview_prompts}")

    history = []
    step = start_step
    running_loss, running_n = 0.0, 0

    for epoch in range(args.epochs):
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            captions = [CAPTION_TEMPLATE.format(train_set.classes[l.item()]) for l in labels]

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)

            optimizer.zero_grad()
            loss = training_step(model, images, captions, args.uncond_prob, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_n += 1
            step += 1

            if step % 50 == 0:
                avg = running_loss / running_n
                print(f"step {step}: loss {avg:.4f} lr {lr_at(step):.2e}")
                history.append({"step": step, "epoch": epoch + 1, "loss": avg})
                running_loss, running_n = 0.0, 0

            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.out_dir, f"minisd_lora_boxcars_{args.part}_step{step}.pt")
                torch.save({
                    "lora_state_dict": lora_state_dict(model.model.diffusion_model),
                    "rank": args.rank, "alpha": args.alpha,
                    "part": args.part, "classes": train_set.classes,
                    "step": step, "epoch": epoch + 1,
                }, ckpt_path)
                print(f"saved {ckpt_path}")

                model.eval()
                previews = sample_images(model, preview_prompts, image_size=args.image_size,
                                         ddim_steps=args.ddim_steps, scale=args.scale,
                                         seed=args.seed, device=device)
                preview_path = os.path.join(args.out_dir, f"preview_step{step}.png")
                save_image(previews, preview_path, nrow=len(preview_prompts))
                print(f"saved {preview_path}")

    final_path = os.path.join(args.out_dir, f"minisd_lora_boxcars_{args.part}_final.pt")
    torch.save({
        "lora_state_dict": lora_state_dict(model.model.diffusion_model),
        "rank": args.rank, "alpha": args.alpha,
        "part": args.part, "classes": train_set.classes,
        "step": step, "epoch": args.epochs,
    }, final_path)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"done -> {final_path}")
    print("next: python evaluate_minisd_lora.py --root <boxcars root> "
         f"--lora-path {final_path} --out-dir {args.out_dir}/eval")


if __name__ == "__main__":
    sys.exit(main())
