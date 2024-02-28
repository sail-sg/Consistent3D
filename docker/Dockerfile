# Copyright 2024 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/pytorch:22.08-py3
USER root

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set compute capability for nerfacc and tiny-cuda-nn
# See https://developer.nvidia.com/cuda-gpus and limit number to speed-up build
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
# ENV TCNN_CUDA_ARCHITECTURES=90;89;86;80;75;70;61;60
# Speed-up build for RTX 30xx
# ENV TORCH_CUDA_ARCH_LIST="8.6"
# ENV TCNN_CUDA_ARCHITECTURES=86
# Speed-up build for RTX 40xx
# ENV TORCH_CUDA_ARCH_LIST="8.9"
# ENV TCNN_CUDA_ARCHITECTURES=89
# Speed-up build for A100
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV TCNN_CUDA_ARCHITECTURES=80

# apt install by root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools ninja
# Install nerfacc and tiny-cuda-nn before installing requirements.txt
# because these two installations are time consuming and error prone
RUN pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch

COPY requirements.txt /tmp
RUN cd /tmp && pip install -r requirements.txt

RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN pip install git+https://github.com/sail-sg/Adan.git

WORKDIR /workspace