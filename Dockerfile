# From pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
From pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install --upgrade pip

RUN apt install -y git
RUN apt-get -y install cmake
RUN apt install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/CompVis/taming-transformers.git
ENV PYTHONPATH "${PYTHONPATH}:./taming-transformers"

RUN pip3 install -e git+https://github.com/openai/CLIP.git@main#egg=clip

RUN pip3 install git+https://github.com/lmb-freiburg/ldce.git
RUN git clone https://github.com/lmb-freiburg/ldce.git
# RUN pip3 install ./ldce/ldm

# RUN pip3 install git+https://github.com/rachtibat/zennit-crp

RUN pip3 install certifi
COPY ca-certificates /usr/local/share/ca-certificates
RUN apt-get install --yes --no-install-recommends software-properties-common ca-certificates
# RUN chmod 644 /usr/local/share/ca-certificates/continental.crt
# RUN chmod 644 /usr/local/share/ca-certificates/conti-corp-it-security.crt
RUN update-ca-certificates

RUN pip3 install corelay[umap,hdbscan]
RUN pip3 install opencv-python
RUN pip3 install seaborn
# RUN pip3 install scipy

# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_DIR=/etc/ssl/certs

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
# CMD ["python", "run_ldce_baseline.py", "--config-name=v1_concept_vit"]
# CMD ["python", "run_concept_ldce.py", "--config-name=v1_concept"]

# CMD ["python", "run_concept_ldce.py", "--config-name=v1_pets"]

# CMD ["python", "src/evaluation/compute_fid.py", "--output-path=/results/counterfactuals/imagenet_resnet_baseline"]
# CMD ["python", "src/evaluation/compare_concept_activations.py", "--config-name=v1_concept"]
# CMD ["python", "src/evaluation/obtain_wrong_predictions.py", "--config-name=v1_vgg_concept"]
# CMD ["python", "run_evaluation.py", "--output-path=/results/counterfactuals/imagenet_vgg16bn_baseline_attrtarget_optim"]

# CMD ["python", "src/clustering/compute_clustering.py", "--config-name=v1_cluster"]
# CMD ["python", "src/concept_analysis.py", "--config-name=v1_concept"]
# CMD ["python", "src/visualization/show_gradient_alignment.py"]
# CMD ["python", "src/write_conditioning.py", "--config-name=v1_concept"]
# CMD ["python", "src/visualization/visualize_localization.py", "--config-name=v1_concept"]

# CMD ["python", "run_ldce.py", "--config-name=v1_wider"]
# CMD ["python", "src/encode_dataset.py", "--config-name=v1_cluster_vit"]
# CMD ["python", "src/encode_counterfactuals.py", "--config-name=v1_cluster"]
# CMD ["python", "src/encode_actattr.py", "--config-name=v1_cluster"]

# CMD ["python", "src/concept_maximization.py", "--config-name=v1_wider"]
# CMD ["python", "src/target_selection.py", "--config-name=v1_concept_vit"]
# CMD ["python", "src/sample_visualization.py", "--config-name=v1_wider"]

# CMD ["python", "src/visualization/visualizer.py", "--config-name=v1_wider"]
# CMD ["python", "src/vit/visualize_concepts.py", "--config-name=v1_concept_vit"]

# CMD ["python", "convert_imagenet.py"]
# CMD ["python", "test_stuff.py", "--config-name=v1_wider"]
# CMD ["python", "src/training/fine_tune_pets.py"]
# CMD ["python", "src/training/fine_tune_flowers.py"]
# CMD ["python", "src/evaluation/print_targets.py"]

# CMD ["python", "run_feature_optim.py", "--config-name=v1_dreamer"]
CMD ["python", "src/visualization/show_num_concepts.py"]
# CMD ["python", "src/visualization/show_num_concept_example.py"]
# CMD ["python", "src/visualization/show_explanations.py", "--config-name=v1_concept"]
# CMD ["python", "src/visualization/show_spatial_constraint.py"]
# CMD ["python", "src/visualization/show_validity.py"]