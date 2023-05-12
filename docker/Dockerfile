# base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# set timezone
ENV  TZ=Asia/Shanghai

# update system & tools
RUN  apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt -y install tzdata \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    git-lfs \
    vim \
    libopenmpi-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    sudo \
    pwgen \
    passwd

# clean file system
RUN apt-get clean \
    && sudo apt-get autoclean \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo rm -rf /root/.cache/pip \
    && sudo rm -rf /var/cache/apt/archives/*


# add user & group & set uid & gid
RUN  groupadd -g 8001  lk && useradd -u 8001 -r -g lk lk \
    && mkdir /home/lk  \
    && chown lk:lk /home/lk \
    && usermod -aG sudo lk  \
    && echo "lk:password@lk" | chpasswd


# switch to lk
USER lk

# add path
ENV PATH="/home/lk/.local/bin:${PATH}"

# set global index & upgrade pip & install torch + cu118
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    && mkdir -p ~/.pip \
    && pip3 install --upgrade pip \
    && pip3 install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# install python packages
RUN pip3 install \
    transformers \
    jsonargparse \
    numpy \
    datasets \
    sentencepiece \
    bitsandbytes \
    accelerate \
    cpm_kernels \
    gradio \
    fastapi uvicorn sse-starlette asyncio \
    tenacity \
    jaxlib==0.3.22 \
    tiktoken \
    triton \
    deepspeed \
    deep_training


