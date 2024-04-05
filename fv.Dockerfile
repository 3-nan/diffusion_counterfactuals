# From pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
From pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip

RUN apt install -y git
RUN apt-get -y install cmake
RUN apt install build-essential -y

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install certifi
COPY ca-certificates /usr/local/share/ca-certificates
RUN apt-get install --yes --no-install-recommends software-properties-common ca-certificates
# RUN chmod 644 /usr/local/share/ca-certificates/continental.crt
# RUN chmod 644 /usr/local/share/ca-certificates/conti-corp-it-security.crt
RUN update-ca-certificates

RUN pip3 install corelay

RUN pip3 install git+https://github.com/rachtibat/zennit-crp.git
RUN git clone https://github.com/lmb-freiburg/ldce.git

# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs

ENV HYDRA_FULL_ERROR=1

RUN export HDF5_USE_FILE_LOCKING='FALSE'

# COPY ./ldce/ldm/ ./ldm

COPY configs/ ./configs
COPY data/ ./data
COPY models/ ./models
COPY run_ldce_baseline.py ./run_ldce_baseline.py
COPY src/ ./src
COPY crp/ ./crp

# CMD ["python", "crp/run_feature_visualization.py"]
CMD ["python", "src/visualization/visualize_concepts.py", "--config-name=v1_concept"]
# CMD ["python", "src/visualization/visualize_conditioning.py", "--config-name=v1_concept"]
