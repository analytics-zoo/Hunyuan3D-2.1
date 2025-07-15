# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ======== Base Stage ========
FROM intel/deep-learning-essentials:2025.0.2-0-devel-ubuntu24.04 AS vllm-base

ARG https_proxy
ARG http_proxy

# Add Intel oneAPI repo and PPA for GPU support
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics-testing

# Install dependencies and Python 3.10
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -y && \
    apt-get install -y python3.10 python3.10-distutils python3.10-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get install -y --no-install-recommends --fix-missing \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libgl1 \
        lsb-release \
        numactl \
        wget \
        vim \
        linux-libc-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    # Install Intel GPU runtime packages
    apt-get update -y && \
    apt-get install -y libze1 libze-dev libze-intel-gpu1 intel-opencl-icd libze-intel-gpu-raytracing && \
    apt-get install -y intel-oneapi-dpcpp-ct=2025.0.1-17 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /llm
COPY ./patches/0001-oneccl-align-global-V0.1.1.patch /tmp/

# Set environment variables early
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

# ======= Add oneCCL build =======
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Build 1ccl
RUN git clone https://github.com/oneapi-src/oneCCL.git && \
    cd oneCCL && \
    git checkout def870543749186b6f38cdc865b44d52174c7492 && \
    git apply /tmp/0001-oneccl-align-global-V0.1.1.patch && \
    mkdir build && cd build && \
    export IGC_VISAOptions=-activeThreadsOnlyBarrier && \
    /usr/bin/cmake .. \
        -DCMAKE_INSTALL_PREFIX=_install \
        -DCMAKE_C_COMPILER=icx \
        -DCMAKE_CXX_COMPILER=icpx \
        -DCOMPUTE_BACKEND=dpcpp \
        -DCCL_ENABLE_ARCB=1 && \
    make -j && make install && \
    mv _install /opt/intel/oneapi/ccl/2021.15.3 && \
    cd /opt/intel/oneapi/ccl/ && \
    ln -snf 2021.15.3 latest


ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/open3d:$LD_LIBRARY_PATH"

RUN pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu && \
    pip install bigdl-core-xe-all==2.7.0b20250625 && \
    apt remove python3-blinker -y
    
RUN git clone -b test_0625 https://github.com/xiangyuT/Hunyuan3D-2.1.git && \
    cd Hunyuan3D-2.1 && \
    pip install -r requirements.txt && \
    cd hy3dpaint/custom_rasterizer && \
    pip install --upgrade setuptools>=64 && \
    export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH && \
    export CPATH=/usr/include:/usr/local/lib/python3.10/dist-packages/torch/include/:$CPATH && \
    python3 setup.py install && \
    cd ../.. && \
    cd hy3dpaint/DifferentiableRenderer && \
    bash compile_mesh_painter.sh && \
    cd ../.. && \
    wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt

WORKDIR /llm/Hunyuan3D-2.1
