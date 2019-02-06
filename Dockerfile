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
    && git clone --single-branch --depth 1 https://github.com/seung-lab/cloud-volume.git \
    && pip install --user --upgrade pip \
    && pip install --no-cache-dir -r /root/cloud-volume/requirements.txt \
    && pip install -r /root/chunkflow/requirements.txt --no-cache-dir \
    # clean up apt install
    && apt-get clean \
    && apt-get autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    # setup environment variables
    && echo "export LC_ALL=C.UTF-8" >> /root/.bashrc \
    && echo "export LANG=C.UTF-8" >> /root/.bashrc \
    && echo "export PYTHONPATH=/root/chunkflow:/root/cloud-volume:\$PYTHONPATH" >> /root/.bashrc

WORKDIR /root/chunkflow/scripts
