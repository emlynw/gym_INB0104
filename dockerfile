# Use the official PyTorch Docker image with CUDA support
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
# FROM nvcr.io/nvidia/pytorch:24.10-py3

# Install system dependencies
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglew2.1 || apt-get install -y libglew-dev
RUN apt-get update && apt-get install -y patchelf
RUN apt-get update && apt-get install -y libosmesa6-dev
RUN apt-get update && apt-get install -y wget unzip && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libglfw3 libgl1-mesa-glx libosmesa6


# Install Python packages
RUN pip install --no-cache-dir \
    mujoco \
    gymnasium==0.29.1 \
    tensorboard \
    wandb \
    termcolor \
    PyOpenGL-accelerate \
    opencv-python-headless \
    dm-robotics-transformations \
    scipy \
    scikit-image \
    numpy \
    imageio-ffmpeg

# Upgrade hydra-core
RUN pip install hydra-core --upgrade
RUN pip install hydra-submitit-launcher --upgrade

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV WANDB_API_KEY=758ee79ba7db46c1c5ff265f257692cbcb7d0fc2
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility
