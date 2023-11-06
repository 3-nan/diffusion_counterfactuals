From pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip

RUN apt install -y git

RUN pip3 install -r requirements.txt

RUN git clone https://github.com/lmb-freiburg/ldce.git

COPY configs/ ./configs
COPY run_ldce.py .run_ldce.py

RUN ["python", "-m run_ldce --config-name=v1_wider data.batch_size=5 strength=0.382 data.start_sample=$id data.end_sample=$((id+1)) > logs/imagenet_sd_${id}.log"]