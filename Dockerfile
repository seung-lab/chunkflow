# backend: pytorch | pznet
ARG BACKEND=pytorch

FROM seunglab/chunkflow:${BACKEND}


LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

WORKDIR /root 
RUN mkdir chunkflow
ADD . chunkflow/ 

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
    apt-utils \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -U pip --no-cache-dir

ENV PYTHONPATH /root/cloud-volume:$PYTHONPATH
ENV PYTHONPATH /root/chunkflow:$PYTHONPATH

# use my own branch to fix some version conflicts
RUN git clone --depth 1 --single-branch -b jingpengw-patch-1 https://github.com/seung-lab/cloud-volume.git \
    && pip install --upgrade setuptools numpy  --no-cache-dir \ 
    && pip install -r /root/cloud-volume/requirements.txt --no-cache-dir \
    && pip install -r /root/chunkflow/requirements.txt --no-cache-dir
