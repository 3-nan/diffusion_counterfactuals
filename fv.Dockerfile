# Separate image for CRP feature visualization (crp/run_feature_visualization.py,
# src/visualization/visualize_concepts.py).
#
# zennit-crp 0.6.0 declares torch<2.0.0, zennit<=0.4.6, numpy<=1.23.5 (see
# `pip show zennit-crp`). Installing it straight into vehicles.Dockerfile's
# environment (confirmed live, the hard way) makes pip silently downgrade
# torch 2.1.0 -> 1.13.1, which then conflicts with kornia/accelerate (both
# require torch>=2.0) and drags in a torchvision build compiled against a
# different CUDA major version than the downgraded torch, breaking
# torchvision.extension at import time. Hence a separate image.
#
# This was originally `FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime`,
# but Docker Hub pulls aren't reliably available from this host (see
# vehicles.Dockerfile's own note on this -- same class of problem, confirmed
# again here: `podman pull pytorch/pytorch:1.11.0-...` fails with
# "authentication required" fetching the blob). Building on top of the
# already-built `vehicles` image instead needs no fresh pull, already has
# everything CRP's own code needs at import time (run_ldce_baseline.py pulls
# in ldce/timm/pytorch_lightning etc. just by being imported, even though
# run_feature_visualization.py only calls get_classifier/get_dataset from
# it), and --no-deps below keeps CRP's stale upper-bound pins from ever
# touching torch/zennit/numpy -- CRP's actual code is plain hook-based
# forward/backward attribution, nothing torch-2.x-specific breaks it, only
# its packaged metadata's bounds are stale (never bumped upstream).
FROM localhost/vehicles:latest

RUN pip3 install --no-cache-dir --no-deps "git+https://github.com/rachtibat/zennit-crp.git"

# No COPY here on purpose -- same reasoning as vehicles.Dockerfile: bind-mount
# the repo at runtime (`-v $(pwd):/workspace`, or via podman as vehicles-dev/2
# already do) rather than baking files in, so editing
# crp/run_feature_visualization.py doesn't require a rebuild.
