# backend: pytorch | pznet
ARG BACKEND=pytorch

FROM seunglab/chunkflow:${BACKEND}


LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"


RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        curl \
        wget \
        git \
        libboost-dev \
        build-essential \
    && cd / \
    && git clone --single-branch --depth 1 https://github.com/seung-lab/igneous.git \
    && cd /igneous \
    && pip install --user --upgrade pip \
    && pip install -U setuptools \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . \
    # Cleanup build dependencies
    && apt-get remove --purge -y \
        libboost-dev \
        build-essential \
    && apt-get autoremove --purge -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 


ENV PYTHONPATH /root/chunkflow:$PYTHONPATH
WORKDIR /root 
RUN mkdir chunkflow
ADD . chunkflow/ 
RUN pip install -r /root/chunkflow/requirements.txt --no-cache-dir \
    && rm -rf /var/lib/apt/lists/*
