# backend: pytorch | pznet
ARG BACKEND=pytorch

FROM seunglab/chunkflow:${BACKEND}

LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

ENV PYTHONPATH /root/cloud-volume:$PYTHONPATH
ENV PYTHONPATH /root/chunkflow:$PYTHONPATH
WORKDIR /root 
RUN mkdir chunkflow
ADD . chunkflow/ 

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        wget \
        git \
    && cd /root \
    && git clone --single-branch --depth 1 -b jwu-numpy-io https://github.com/seung-lab/cloud-volume.git \
    && pip install --user --upgrade pip \
    && pip install --no-cache-dir -r /root/cloud-volume/requirements.txt \
    && apt-get autoremove --purge -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \ 
    && pip install -r /root/chunkflow/requirements.txt --no-cache-dir \
    && rm -rf /var/lib/apt/lists/*
