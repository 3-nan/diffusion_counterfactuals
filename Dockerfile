From pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip

RUN apt install -y git

RUN pip3 install -r requirements.txt