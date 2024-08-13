FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential python3-opencv dos2unix

RUN pip3 install --upgrade pip==24.0  # ==24.0

RUN apt install -y git
RUN apt-get -y install cmake
# RUN apt install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install numpy==1.24.1 opencv-python timm cub-tools --no-cache-dir

COPY src/ ./src

# CMD ["python", "src/training/fine_tune_cubs.py", "--config-name=v1_cubs"]
CMD ["python", "src/training/fine_tune_adience.py"]