# SD3.5 Medium counterfactual generation -- a thin, additive layer on top of
# the already-built and already-working `vehicles` image (vehicles.Dockerfile),
# not a modification of it. Kept as a separate Dockerfile/image specifically
# so this stays isolated: vehicles.Dockerfile / vehicles-dev / vehicles-dev2
# are untouched, and if anything about the SD3.5 pin below turns out to be
# wrong it can't destabilize the working miniSD/Cars pipeline.
#
# Why this needs its own diffusers pin at all: `pip3 install "diffusers>=0.31.0"`
# in vehicles.Dockerfile resolves to whatever the latest release is at build
# time (0.39.0 as of this writing), and current diffusers unconditionally
# references `torch.xpu.empty_cache` in utils/torch_utils.py at import time --
# `torch.xpu` doesn't exist until torch 2.4+, so `import diffusers` itself
# hard-fails against this repo's torch==2.1.0 pin. This was never caught
# because nothing had actually imported diffusers yet (the SD1.5-via-diffusers
# route in src/diffusers/ was also never run end-to-end). Re-pinning to
# 0.31.0 -- the same floor version vehicles.Dockerfile's comment already
# named as the minimum for StableDiffusion3Img2ImgPipeline support -- avoids
# the torch.xpu reference and was verified live in this exact image
# (`import diffusers` succeeds, `from diffusers import
# StableDiffusion3Img2ImgPipeline` succeeds).
FROM localhost/vehicles:latest

RUN pip3 install --no-cache-dir --no-deps "diffusers==0.31.0"

# Guardrail, same pattern as vehicles.Dockerfile's torch/torchvision/numpy
# check: fail the build loudly here rather than downstream mid-generation if
# some future change lets diffusers drift back to a torch.xpu-requiring
# version.
RUN python3 -c "\
import diffusers; \
from diffusers import StableDiffusion3Img2ImgPipeline, SD3Transformer2DModel; \
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler; \
assert diffusers.__version__ == '0.31.0', f'diffusers drifted to {diffusers.__version__}'; \
print(f'diffusers {diffusers.__version__} OK, StableDiffusion3Img2ImgPipeline importable')"

ENV HYDRA_FULL_ERROR=1

# No COPY -- same bind-mount pattern as vehicles.Dockerfile, see the notes
# there. WORKDIR is inherited from vehicles:latest (/workspace).

# ---------------------------------------------------------------------------
# Usage
#
#   podman build -t vehicles-sd3 -f vehicles-sd3.Dockerfile .
#   podman run -d --name vehicles-sd3-dev --shm-size=8g --gpus all \
#       -v $(pwd):/workspace \
#       -v /rbcn007x/share/franz/StanfordCars:/data/stanford_cars \
#       -v /rbcn004x/share/franz/counterfactuals/results:/results \
#       -e HF_TOKEN=... \
#       vehicles-sd3 sleep infinity
#
# SD3.5 Medium weights are gated: accept the license at
# https://huggingface.co/stabilityai/stable-diffusion-3.5-medium first, then
# either pass -e HF_TOKEN=... (read at from_pretrained() call time) or run
# `huggingface-cli login` inside the container once. First run downloads and
# caches a local snapshot at /results/models/stable-diffusion-3.5-medium
# (see run_diffusers.py's `sd3` branch); subsequent runs load from there
# without needing the token again.
#
#   podman exec -it -e CUDA_VISIBLE_DEVICES=<free> -e HF_TOKEN=... \
#       vehicles-sd3-dev python src/diffusers/run_diffusers.py --config-name=v1_cars_sd3
# ---------------------------------------------------------------------------
