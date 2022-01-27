# BSD 3-Clause License
#
# Copyright (c) 2022, Dan Novischi
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ##################################################################################
# Setup Nvidia CUDA for Jetson Nano
# ##################################################################################
ARG V_OS_MAJOR=20
ARG V_OS_MINOR=04
ARG V_OS=${V_OS_MAJOR}.${V_OS_MINOR}
FROM arm64v8/ubuntu:20.04 as jetson-cuda
# Configuration Arguments
ARG TIMEZONE=Europe/Bucharest
ARG V_SOC=t210
ARG V_CUDA_MAJOR=10
ARG V_CUDA_MINOR=2
ARG V_L4T_MAJOR=32
ARG V_L4T_MINOR=6
ENV V_CUDA=${V_CUDA_MAJOR}.${V_CUDA_MINOR}
ENV V_CUDA_DASH=${V_CUDA_MAJOR}-${V_CUDA_MINOR}
# ENV V_L4T=r${V_L4T_MAJOR}.${V_L4T_MINOR}
ENV V_L4T=r${V_L4T_MAJOR}.${V_L4T_MINOR}
# Expose environment variables everywhere
ENV CUDA=${V_CUDA_MAJOR}.${V_CUDA_MINOR}
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Fix CUDA info
ARG DPKG_STATUS
# Set timezone
ENV TZ=${TIMEZONE}
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
# Add NVIDIA repo/public key and install VPI libraries
RUN echo "$DPKG_STATUS" >> /var/lib/dpkg/status \
    && echo "[Builder] Installing Prerequisites" \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates software-properties-common curl gnupg2 apt-utils \
    ninja-build git cmake libjpeg-dev libopenmpi-dev libomp-dev ccache\
    libopenblas-dev libblas-dev libeigen3-dev python3-pip

RUN echo "[Builder] Installing CUDA Repository" \
    && curl https://repo.download.nvidia.com/jetson/jetson-ota-public.asc > /etc/apt/trusted.gpg.d/jetson-ota-public.asc \
    && echo "deb https://repo.download.nvidia.com/jetson/common ${V_L4T} main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "deb https://repo.download.nvidia.com/jetson/${V_SOC} ${V_L4T} main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
    && echo "[Builder] Installing CUDA System" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-libraries-${V_CUDA_DASH} \
    cuda-libraries-dev-${V_CUDA_DASH} \
    cuda-nvtx-${V_CUDA_DASH} \
    cuda-minimal-build-${V_CUDA_DASH} \
    cuda-license-${V_CUDA_DASH} \
    cuda-command-line-tools-${V_CUDA_DASH} \
    nvidia-cudnn* \
    libnvvpi1 vpi1-dev \
    && ln -s /usr/local/cuda-${V_CUDA} /usr/local/cuda \
    && rm -rf /var/lib/apt/lists/*
# ##################################################################################
# Create PyTorch Download Layer
# We do this seperately since else we need to keep rebuilding
# ##################################################################################
FROM arm64v8/ubuntu:20.04 as download
# Set timezone
ENV TZ=Europe/Bucharest
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone
# Configuration Arguments
# https://github.com/pytorch/pytorch
ARG V_PYTORCH=v1.10.0
# https://github.com/pytorch/vision
ARG V_TORCHVISION=v0.11.1
# https://github.com/pytorch/audio
ARG V_TORCHAUDIO=v0.10.0
# Install Git Tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common apt-utils git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Clone Source
RUN git clone --recursive --branch ${V_PYTORCH} http://github.com/pytorch/pytorch
RUN git clone --recursive --branch ${V_TORCHVISION} https://github.com/pytorch/vision.git
RUN git clone --recursive --branch ${V_TORCHAUDIO} https://github.com/pytorch/audio.git
# ##################################################################################
# Build PyTorch for Jetson (with CUDA)
# ##################################################################################
FROM jetson-cuda as build
# Configuration Arguments
ARG V_PYTHON_MAJOR=3
ARG V_PYTHON_MINOR=8
ENV V_CLANG=8
ENV V_PYTHON=${V_PYTHON_MAJOR}.${V_PYTHON_MINOR}
# Accept default answers for everything
ENV DEBIAN_FRONTEND=noninteractive
# Download Common Software
RUN apt-get update \
    && apt-get install -y clang clang-${V_CLANG} build-essential bash ca-certificates git wget cmake curl software-properties-common ffmpeg libsm6 libxext6 libffi-dev libssl-dev xz-utils zlib1g-dev liblzma-dev \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${V_CLANG} 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${V_CLANG} 100 \
    && update-alternatives --set clang /usr/bin/clang-${V_CLANG} \
    && update-alternatives --set clang++ /usr/bin/clang++-${V_CLANG}

# Setting up Python
WORKDIR /install
RUN if [ "$V_PYTHON" != "3.6" ] && [ "$V_PYTHON" != "3.8" ]; then \
    add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python${V_PYTHON} python${V_PYTHON}-dev python${V_PYTHON}-venv python${V_PYTHON_MAJOR}-tk \
    && rm /usr/bin/python \
    && rm /usr/bin/python${V_PYTHON_MAJOR} \
    && ln -s $(which python${V_PYTHON}) /usr/bin/python \
    && ln -s $(which python${V_PYTHON}) /usr/bin/python${V_PYTHON_MAJOR} \
    && curl --silent --show-error https://bootstrap.pypa.io/get-pip.py | python; \
    else \
    apt-get update \
    && apt-get install -y python${V_PYTHON} python${V_PYTHON}-dev python${V_PYTHON}-venv python${V_PYTHON_MAJOR}-tk; \
    fi

# PyTorch - Build - Source Code Setup (copy repos from download to build)
COPY --from=download /pytorch /pytorch
COPY --from=download /vision /vision
COPY --from=download /audio /audio

WORKDIR /pytorch
# PyTorch - Build - Prerequisites
# Set clang as compiler
# clang supports the ARM NEON registers
# GNU GCC will give "no expression error"
ARG CC=clang
ARG CXX=clang++
# Set path to ccache
ARG PATH=/usr/lib/ccache:$PATH
# Other arguments
ARG USE_CUDA=ON
ARG USE_CUDNN=ON
ARG BUILD_CAFFE2_OPS=0
ARG USE_FBGEMM=0
ARG USE_FAKELOWP=0
ARG BUILD_TEST=0
ARG USE_MKLDNN=0
ARG USE_NNPACK=0
ARG USE_XNNPACK=0
ARG USE_QNNPACK=0
ARG USE_PYTORCH_QNNPACK=0
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"
ARG USE_NCCL=0
ARG USE_SYSTEM_NCCL=0
ARG USE_OPENCV=0
ARG USE_DISTRIBUTED=0
# PyTorch Build
RUN cd /pytorch \
    && rm -rf build/CMakeCache.txt || : \
    && sed -i -e "/^if(DEFINED GLIBCXX_USE_CXX11_ABI)/i set(GLIBCXX_USE_CXX11_ABI 1)" CMakeLists.txt \
    && pip3 install wheel mock pillow \
    && pip3 install scikit-build \
    && python3 -WORKDIR /pytorch
m pip install setuptools==59.5.0 \
    && pip3 install -r requirements.txt \
    && python3 setup.py bdist_wheel \
    && cd ..

# Install the PyTorch wheel
RUN apt-get install -y libswresample-dev libswscale-dev libavformat-dev libavcodec-dev libavutil-dev \
    && cd /pytorch/dist/ \
    && pip3 install `ls` \
    && cd .. \
    && cd ..

# Torchvision Build
WORKDIR ../vision
RUN cd /vision \
    && python3 setup.py clean \
    && python3 setup.py bdist_wheel \
    && mkdir -p output \
    && cp dist/*.whl output/ \
    && cd ..

# # Torchaudio Build
# WORKDIR /audio
# RUN cd /audio \
#     && python3 -m pip install setuptools==59.5.0 \
#     && python3 setup.py bdist_wheel \
#     && cd ..

# # ##################################################################################
# # Prepare Artifact
# # ##################################################################################
FROM scratch as artifact
COPY --from=build /pytorch/dist/* /
# COPY --from=build /vision/dist/* /

# COPY --from=build /audio/dist/* /
