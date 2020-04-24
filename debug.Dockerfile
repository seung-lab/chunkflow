# backend: base | pytorch | pznet | pytorch-cuda9
ARG BACKEND=pytorch

FROM seunglab/chunkflow:${BACKEND}

LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

RUN mkdir -p $HOME/workspace/chunkflow

# WORKDIR only works with ENV     

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV HOME /root

WORKDIR $HOME/workspace/chunkflow

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        wget \
        git \
        build-essential \
        python3-dev \
        parallel \
        ca-certificates \
        gnupg-agent \
        gnupg \
        dirmngr \
    # test whether pip is working 
    # there is an issue of pip:
    # https://github.com/laradock/laradock/issues/1496
    # we need this hash to solve this issue
    # && ln -sf /usr/bin/pip3 /usr/bin/pip \
    # this do not work due to an issue in pip3
    # https://github.com/pypa/pip/issues/5240
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y \
    && pip install -U pip \
    && hash -r pip \
    && pip install --upgrade setuptools \
    && pip install numpy setuptools cython tornado python-dateutil --no-cache-dir \
    # && pip install fpzip --no-binary :all: --no-cache-dir \
    # && git clone --single-branch --depth 1 https://github.com/seung-lab/cloud-volume.git \
    # && pip install --no-cache-dir -r $HOME/workspace/cloud-volume/requirements.txt \
    # setup environment variables 
    # we have to setup first, otherwise click installation will fail
    && echo "[GoogleCompute]\nservice_account = default" > /etc/boto.cfg \
    && echo "export LC_ALL=C.UTF-8" >> $HOME/.bashrc \
    && echo "export LANG=C.UTF-8" >> $HOME/.bashrc \
    && echo "export PYTHONPATH=$HOME/workspace/chunkflow:\$PYTHONPATH" >> $HOME/.bashrc

COPY . $HOME/workspace/chunkflow

RUN pwd && ls \
    && pip install -r requirements.txt --no-cache-dir \
    && pip install -r tests/requirements.txt --no-cache-dir \
    # install the commandline chunkflow
    && pip install -e . \
    && git clone https://github.com/seung-lab/DeepEM \
    && git clone https://github.com/seung-lab/dataprovider3 \
    && git clone https://github.com/seung-lab/pytorch-emvision \
    # cleanup system libraries 
    # the test will not pass due to missing of credentials.
    # && pytest tests \
    && chunkflow

WORKDIR $HOME/workspace/
