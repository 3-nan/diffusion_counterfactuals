# Image for the vehicle-dataset experiments: finetune the classifier here,
# then run counterfactual generation against either backbone from the same
# image -- miniSD/SD1.5 (torchvision.models.vgg-style img2img, near-zero
# porting cost) and SD3.5 Medium (MMDiT + rectified flow, 16-channel VAE).
#
# This deliberately does NOT install from requirements.txt. That file pins
# transformers==4.31.0 / huggingface-hub==0.16.4 / pytorch-lightning==1.4.2 /
# numpy==1.21.5 for the classic pytorch 1.13-era ldm stack, which conflicts
# with the modern diffusers/transformers SD3.5 needs. diffusers.Dockerfile
# already sidesteps this the same way for the img2img route -- see there.
# `pip install git+.../ldce.git` still pulls in pytorch_lightning itself
# (src/diffusers/run_diffusers.py imports ldce.sampling_helpers, which needs
# it to instantiate the CompVis LatentDiffusion class even for inference),
# so let its own setup.py pick a version compatible with this torch instead
# of hand-pinning the ancient one.
#
# Driver on this host is 580.95 / CUDA 13.0, so cuda11.8 wheels run fine via
# forward compatibility -- bump if you move to a host with an older driver.
#
# Base is localhost/glca:latest, NOT pytorch/pytorch directly, because Docker
# Hub pulls aren't available from this machine and getting apt/pip working
# from scratch here needs more than the CA cert -- glca's build history
# (`podman history --no-trunc localhost/glca:latest`) shows the actual fix:
# archive.ubuntu.com/security.ubuntu.com must be rewritten from http:// to
# https:// in /etc/apt/sources.list, or the transparent proxy on this network
# redirects into a captive-portal auth loop that apt/curl can't complete
# non-interactively. Trusting the corporate CA alone (what we tried first)
# gets you a clean TLS handshake but not past that. glca already carries that
# sed fix plus the corporate CA trusted at both the system and certifi level,
# and already has torch==2.1.0 / torchvision==0.16.0 / transformers==4.45.2 /
# scipy / zennit installed, so we build on top instead of duplicating it.
#
# Caveat: glca belongs to a different (GroundingDINO/SAM) project on this
# host and isn't something this repo controls -- if it's ever removed or
# rebuilt out from under this, switch back to `FROM pytorch/pytorch:2.1.0-
# cuda11.8-cudnn8-runtime` and reapply the sed fix above (apt-get update
# needs to work at all first, so do the CA-cert COPY block from git history
# on this file, then `sed -i 's|http://\(archive\|security\).ubuntu.com|
# https://\1.ubuntu.com|g' /etc/apt/sources.list` before the first apt-get
# update).
FROM localhost/glca:latest

ENV DEBIAN_FRONTEND=noninteractive
# glca's own cache namespace / import path -- not ours.
ENV TORCH_HOME=/results/vehicles/torch
# Persists the HF cache (CLIPTextModel.from_pretrained, etc.) under the same
# /results mount instead of the container's ephemeral filesystem -- without
# this it silently redownloads every time the container gets recreated
# (which happens a lot -- every image rebuild in this Dockerfile's iteration
# needs a fresh container).
ENV HF_HOME=/results/vehicles/hf_cache
ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y build-essential python3-opencv dos2unix cmake

# ---------------------------------------------------------------------------
# conda's Python ignores SSL_CERT_FILE/SSL_CERT_DIR -- its OpenSSL is built
# with a hardcoded cafile (`python3 -c "import ssl;
# print(ssl.get_default_verify_paths())"` shows openssl_cafile=/opt/conda/
# ssl/cert.pem regardless of the env vars). curl/pip/requests-based tools
# work fine (they honor CURL_CA_BUNDLE / REQUESTS_CA_BUNDLE, or use certifi),
# but stdlib urllib doesn't -- which is what torch.hub uses for pretrained-
# weight downloads. Symlinking straight to the system bundle (not appending
# our two raw source cert files -- tried that, still failed; the properly
# assembled /etc/ssl/certs/ca-certificates.crt is the one thing on this host
# actually proven to verify, e.g. via curl) fixes every stdlib-urllib-based
# downloader in the image, including huggingface_hub's occasional urllib
# fallback paths for the SD3.5 weights later.
#
# This does NOT fix torchvision.models.vgg16_bn(pretrained=True) specifically
# -- download.pytorch.org itself is fine, but it 303-redirects to a CDN edge
# that this network's gateway intercepts with a captive-portal auth page
# (sia-transparent-auto...:7831/.../cpauth, url-category rule, not a cert
# problem) that only completes via an interactive browser. No amount of
# cert-trust setup in the container gets past that -- download the checkpoint
# by hand from https://download.pytorch.org/models/vgg16_bn-6c64b313.pth and
# drop it at $TORCH_HOME/hub/checkpoints/vgg16_bn-6c64b313.pth instead.
# huggingface.co itself is NOT behind this block (verified with curl), so
# this is asserted against HF rather than pytorch.org.
# ---------------------------------------------------------------------------
RUN ln -sf /etc/ssl/certs/ca-certificates.crt "$(python3 -c 'import ssl; print(ssl.get_default_verify_paths().openssl_cafile)')" && \
    ln -sf /etc/ssl/certs/ca-certificates.crt "$(python3 -c 'import certifi; print(certifi.where())')" && \
    python3 -c "import warnings, urllib.request; warnings.simplefilter('error'); urllib.request.urlopen('https://huggingface.co', timeout=10); print('stdlib urllib HTTPS OK')"

# ---------------------------------------------------------------------------
# Classifier finetuning deps (finetune_classifier.py, data/datasets.py,
# compute_closest_indices.py, make_vehicle_labelmaps.py). numpy/scipy/pillow/
# pyyaml/tqdm/matplotlib already come from glca -- only pandas/timm/seaborn
# are missing.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir pandas timm seaborn

# ---------------------------------------------------------------------------
# CoLa-DCE / LDCE core: CRP (zennit, already in glca), the ldce package + its
# CompVis ldm checkout, taming-transformers (VQGAN, used by the ldm
# autoencoder), CLIP. corelay is for the concept clustering step
# (src/clustering/). hydra-core/omegaconf/h5py/setuptools/regex already come
# from glca.
#
# numpy<2 constraint: current opencv-python requires numpy>=2 unconditionally
# on python>=3.9, and pip happily resolves that by upgrading numpy -- but
# torch==2.1.0's compiled extensions are built against NumPy 1.x's ABI, which
# breaks torch/numpy interop (torch.from_numpy, .numpy(), etc.) with a
# "_ARRAY_API not found" warning that doesn't fail loudly at import time.
# Pinning numpy<2 here forces pip to resolve back to an opencv-python release
# that still supports it, instead of silently taking the ABI break.
#
# pytorch_lightning: not used for training/inference orchestration anywhere
# in this repo's generation path -- needed purely because miniSD.ckpt (and
# any other CompVis-ldm-era .ckpt) was saved via a Lightning Trainer, and
# torch.load()'s pickle has to resolve every class referenced in the
# checkpoint (optimizer states, PL version metadata, ...) even though
# sampling_helpers.py only reads pl_sd["state_dict"] afterward.
#
# Pinned to 1.4.2 (not latest) on purpose -- this matches requirements.txt's
# original pin exactly, and it's a real requirement, not just caution: the
# vendored ldm code (ldm/models/diffusion/ddpm.py) does
# `from pytorch_lightning.utilities.distributed import rank_zero_only`,
# which only exists pre-1.7 -- latest pytorch_lightning (2.6.5) moved it to
# pytorch_lightning.utilities.rank_zero and fails at import time. torchmetrics
# has to come down with it: pytorch_lightning 1.4.2's own deprecated
# `pytorch_lightning.metrics` shim calls
# `torchmetrics.utilities.data.get_num_classes`, removed from modern
# torchmetrics (glca ships 1.9.0) -- 0.6.0 is what requirements.txt pairs
# with 1.4.2, and nothing else in this repo imports torchmetrics directly, so
# there's nothing else to break by pinning it down. Both verified live
# against this exact torch==2.1.0 image before landing here (`from
# ldm.models.diffusion.ddpm import LatentDiffusion` succeeds).
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir "numpy<2" einops kornia corelay[umap,hdbscan] opencv-python pytorch-msssim lpips pytorch-fid \
    "pytorch_lightning==1.4.2" "torchmetrics==0.6.0" "torch==2.1.0" "torchvision==0.16.0"

# Cloned to /opt, not /workspace: /workspace gets bind-mounted over with the
# host repo at runtime (see the no-COPY note below), which would otherwise
# silently shadow anything cloned into it at build time -- `import ldce`
# works fine in a bare `podman run` (nothing mounted) and breaks the moment
# you add `-v $(pwd):/workspace`, which is exactly the deployment this image
# is for. Absolute PYTHONPATH entries here sidestep that regardless of what
# ends up mounted at /workspace.
RUN git clone https://github.com/CompVis/taming-transformers.git /opt/taming-transformers
ENV PYTHONPATH "${PYTHONPATH}:/opt/taming-transformers"

# NOT -e (editable): an editable install doesn't copy files into
# site-packages, it just points a .pth finder at wherever pip cloned the
# source -- which, same as the ldce/taming-transformers issue above, lands in
# /workspace by default (inherited WORKDIR) and gets shadowed by the runtime
# bind-mount. A normal install actually copies the package in, so it's immune
# to whatever ends up mounted at /workspace. There's no live-editing use case
# for this package here anyway -- editable was never buying us anything.
RUN pip3 install --no-cache-dir "git+https://github.com/openai/CLIP.git@main#egg=clip"

# `pip install git+.../ldce.git` is a no-op left here only as a marker of
# intent: it installs a "latent-diffusion" dist-info with no actual files
# (an upstream packaging gap, not fixable from this Dockerfile) -- the git
# clone below is what actually provides both `ldce.*` and `ldm.*` (ldm lives
# as a subdirectory of the ldce repo itself, so /opt/ldce needs to be on
# PYTHONPATH directly, not just its parent /opt).
RUN pip3 install git+https://github.com/lmb-freiburg/ldce.git
RUN git clone https://github.com/lmb-freiburg/ldce.git /opt/ldce
ENV PYTHONPATH "${PYTHONPATH}:/opt:/opt/ldce"

# ---------------------------------------------------------------------------
# Generation backbones.
#
# miniSD/SD1.5: already covered above (ldce + CLIP + taming-transformers is
# the classic ldm route; src/diffusers/run_diffusers.py's HF-diffusers img2img
# route only needs the diffusers install below).
#
# SD3.5 Medium: MMDiT + rectified flow + 16-channel VAE + T5-XXL text
# encoder. Needs a recent diffusers -- SD3.5 pipeline support landed in
# diffusers 0.31; glca's transformers==4.45.2 and huggingface_hub==0.36.2
# already clear the floor SD3.5 needs, so they're left alone here rather
# than risking pip's resolver bumping/downgrading something against the
# torch==2.1.0 pin. sentencepiece + protobuf are for the T5 tokenizer.
#
# NOT installing bitsandbytes: it would have been for optional 8-bit T5-XXL
# loading (T5-XXL is ~9-10GB in fp16), but current bitsandbytes requires
# torch>=2.4. pip silently satisfied that by upgrading torch to 2.13 here on
# a first pass, which broke torchvision==0.16.0 (its compiled extensions are
# built against torch 2.1's ABI -- `import torchvision` still "succeeds" but
# torchvision.io's C++ extension fails to load). The explicit torch/
# torchvision pins below are a guardrail against that happening silently
# again from some other future package. If you actually need 8-bit T5
# loading, that's a deliberate whole-stack torch>=2.4 upgrade (new base
# image, requalify torchvision/CUDA together), not a one-line addition here.
#
# Gated weights: `stabilityai/stable-diffusion-3.5-medium` requires accepting
# the license on HF and passing a token at *runtime* (`huggingface-cli login`
# or `-e HF_TOKEN=...`), not baked into the image.
# ---------------------------------------------------------------------------
RUN pip3 install --no-cache-dir \
    "diffusers>=0.31.0" "accelerate>=0.34.0" sentencepiece protobuf \
    "torch==2.1.0" "torchvision==0.16.0"

# Guardrail: fail the build here, loudly, if any pip step above let torch /
# torchvision / numpy drift out of the versions this stack was qualified
# against (exactly what bitsandbytes and opencv-python each did above before
# they got pinned/constrained out -- see comments above). Checking the
# version strings alone isn't enough: the numpy break doesn't change
# torch/torchvision's own __version__, it just breaks their compiled
# extensions' ABI against numpy at runtime, and only warns instead of
# raising. So this does a real torch<->numpy roundtrip with warnings
# promoted to errors, not just an import.
RUN python3 -c "\
import warnings, torch, torchvision, torchvision.io, numpy as np; \
assert torch.__version__.startswith('2.1.0'), f'torch drifted to {torch.__version__}'; \
assert torchvision.__version__.startswith('0.16.0'), f'torchvision drifted to {torchvision.__version__}'; \
assert np.__version__.startswith('1.'), f'numpy drifted to {np.__version__} (torch 2.1.0 needs numpy<2)'; \
warnings.simplefilter('error'); \
_ = torch.from_numpy(np.zeros(3, dtype='float32')).numpy(); \
print(f'torch {torch.__version__} / torchvision {torchvision.__version__} / numpy {np.__version__} OK')"

ENV HYDRA_FULL_ERROR=1
RUN export HDF5_USE_FILE_LOCKING='FALSE'

# ---------------------------------------------------------------------------
# No COPY for the repo itself -- mount it at run time instead, so edits to
# finetune_classifier.py / configs / src don't require a rebuild. glca's
# WORKDIR is already /workspace; bind-mount the repo there and every
# relative path below (configs/..., src/diffusers/...) just works:
#
#   docker run --gpus all -it --rm \
#       -v $(pwd):/workspace \
#       -v /rbcn007x/share/franz/StanfordCars:/data/stanford_cars \
#       -v /results:/results \
#       -e HF_TOKEN=... \
#       vehicles python finetune_classifier.py --dataset=cars \
#           --root=/data/stanford_cars --out-dir=/results/models/vgg16bn_cars
#
# checkpoints/ and models/ (gitignored, populated locally) ride along with
# the repo mount too -- no separate volume needed for those.
# ---------------------------------------------------------------------------
WORKDIR /workspace

# Stage 1 -- finetune the classifier you'll explain later:
# python finetune_classifier.py --dataset=cars --root=/data/stanford_cars --out-dir=/results/models/vgg16bn_cars

# Stage 2 -- near-miss targets from the finetuned checkpoint:
# python compute_closest_indices.py --dataset=cars --root=/data/stanford_cars --checkpoint=/results/models/vgg16bn_cars/<ckpt>.pth --out=/data/cars_closest_indices.json

# Stage 3a -- generation on miniSD/SD1.5 (classic ldm route, concept-conditioned):
# python run_concept_ldce.py --config-name=v1_concept

# Stage 3b -- generation on miniSD via the HF-diffusers img2img route:
# python src/diffusers/run_diffusers.py --config-name=v1_diffusers

# Stage 3c -- generation on SD3.5 Medium: no pipeline wired up yet. The
# HF-diffusers route in src/diffusers/pipeline.py (ModifiedStableDiffusion-
# Img2ImgPipeline) is the template -- subclass diffusers'
# StableDiffusion3Img2ImgPipeline the same way, reusing the classifier-
# guidance/concept-masking __call__ override. This image has everything that
# port needs installed; the pipeline code itself doesn't exist yet.

# podman exec -it vehicles-dev python compute_closest_indices.py --dataset=cars \
#     --root=/data/stanford_cars --checkpoint=/results/models/vgg16bn_cars/vgg16_bn_cars_20260727_150956_0.812.pth \
#     --out=/data/cars_closest_indices.json

# podman exec -it -e CUDA_VISIBLE_DEVICES=1 vehicles-dev python run_ldce_baseline.py --config-name=v1_cars

# podman exec -it -e CUDA_VISIBLE_DEVICES=2 vehicles-dev python finetune_minisd_lora.py \
#     --root /data/boxcars --out-dir /results/models/minisd_lora_boxcars

# # baseline for comparison
# podman exec -it -e CUDA_VISIBLE_DEVICES=2 vehicles-dev python evaluate_minisd_lora.py \
#     --root /data/boxcars --out-dir /results/eval/minisd_stock

# # your trained adapter
# podman exec -it -e CUDA_VISIBLE_DEVICES=2 vehicles-dev python evaluate_minisd_lora.py \
#     --root /data/boxcars --lora-path /results/models/minisd_lora_boxcars/minisd_lora_boxcars_hard_final.pt \
#     --out-dir /results/eval/minisd_lora \
#     --classifier-path /results/models/vgg16bn_boxcars/<ckpt>.pth