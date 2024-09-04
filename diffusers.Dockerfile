FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip

RUN apt install -y git
RUN apt-get -y install cmake
RUN apt install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# COPY requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt
RUN pip3 install einops hydra-core h5py kornia omegaconf setuptools timm transformers --no-cache-dir

RUN git clone https://github.com/CompVis/taming-transformers.git
ENV PYTHONPATH "${PYTHONPATH}:./taming-transformers"

RUN pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip

RUN pip3 install git+https://github.com/lmb-freiburg/ldce.git
RUN git clone https://github.com/lmb-freiburg/ldce.git
# RUN pip3 install ./ldce/ldm

# RUN pip3 install git+https://github.com/rachtibat/zennit-crp

RUN pip3 install --upgrade certifi
COPY ca-certificates /usr/local/share/ca-certificates
RUN apt-get install --yes --no-install-recommends ca-certificates
RUN update-ca-certificates

RUN pip3 install corelay[umap,hdbscan] opencv-python seaborn open_clip_torch accelerate diffusers sentencepiece --no-cache-dir

ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs
# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV HYDRA_FULL_ERROR=1

RUN export HDF5_USE_FILE_LOCKING='FALSE'

# COPY ./ldce/ldm/ ./ldm

COPY configs/ ./configs
COPY data/ ./data
COPY models/ ./models
COPY src/ ./src
COPY run_ldce.py ./run_ldce.py
COPY run_ldce_baseline.py ./run_ldce_baseline.py
COPY run_concept_ldce.py ./run_concept_ldce.py
COPY run_feature_optim.py ./run_feature_optim.py
COPY run_evaluation.py ./run_evaluation.py
COPY test_stuff.py ./test_stuff.py

# CMD ["python", "run_ldce_baseline.py", "--config-name=v1_original"]
# CMD ["python", "run_concept_ldce.py", "--config-name=v1_concept"]

# CMD ["python", "run_concept_ldce.py", "--config-name=v1_pets"]
# CMD ["python", "run_concept_ldce.py", "--config-name=v1_flowers"]

# CMD ["python", "src/diffusers/run_diffusers.py", "--config-name=v1_celeba"]
CMD ["python", "src/diffusers/run_diffusers.py", "--config-name=v1_diffusers"]