# The evaluation post-training trainer: a Chalk training-run image whose entrypoint,
# `python -m chalkcompute.training.entrypoint`, calls chalk_harbor.post_training.train_policy.
#
#   docker build -f docker/trainer.Dockerfile -t <registry>/chalk-harbor-trainer:<tag> .
#   docker push <registry>/chalk-harbor-trainer:<tag>
#
# Give the pushed URI as the post-training config's `trainer_image`. The cluster must be able to
# pull from <registry>; scripts/build_trainer_image.py builds the same image with Chalk's image
# builder instead, which pushes it where the cluster pulls from.
FROM python:3.12-slim-bookworm

# The torch wheels bundle CUDA 12.8 (driver >= 570); nothing CUDA is needed from the base image.
ARG TORCH_VERSION=2.9.1
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128

WORKDIR /opt/chalk-harbor
COPY pyproject.toml README.md LICENSE ./
COPY chalk_harbor ./chalk_harbor
# chalkpy provides chalk.ml's checkpoint and log_metrics; polars lets the training entrypoint
# hand the run's dataset to the function.
RUN pip install --no-cache-dir ".[post-training]" chalkpy polars

# Hugging Face downloads (the base model, ~8 GB for a 4B model) go to the container's disk.
# GKE mounts the node's driver libraries (libcuda) under /usr/local/nvidia but adds them to no
# loader path; without LD_LIBRARY_PATH torch finds no GPU there.
ENV HF_HOME=/tmp/hf \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false
