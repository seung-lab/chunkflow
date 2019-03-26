# backend: base | pytorch | pznet 
ARG BACKEND=base

FROM seunglab/chunkflow:${BACKEND}

LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

RUN mkdir -p $HOME/workspace/chunkflow

# WORKDIR only works with ENV 
ENV HOME /root
WORKDIR $HOME/workspace/chunkflow
COPY . .

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        wget \
        git \
        build-essential \
		python3-dev \
        parallel \
    # test whether pip is working 
    # there is an issue of pip:
    # https://github.com/laradock/laradock/issues/1496
	# we need this hash to solve this issue
    # && ln -sf /usr/bin/pip3 /usr/bin/pip \
    # this do not work due to an issue in pip3
    # https://github.com/pypa/pip/issues/5240
    && pip3 install -U pip \
    && hash -r pip \
    && pip3 install --upgrade setuptools \
    && pip3 install numpy setuptools tornado==5.0 --no-cache-dir \ 
    && pip3 install fpzip --no-binary :all: --no-cache-dir \
    # && git clone --single-branch --depth 1 https://github.com/seung-lab/cloud-volume.git \
    # && pip install --no-cache-dir -r $HOME/workspace/cloud-volume/requirements.txt \
    && pip3 install -r requirements.txt --no-cache-dir \
    # install the commandline chunkflow
    && pip3 install -e . \
    # cleanup build dependencies 
    && apt-get remove --purge -y  \
		build-essential \
		python3-dev \
    # clean up apt install
    && apt-get clean \
    && apt-get autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    # setup environment variables
    && echo "export LC_ALL=C.UTF-8" >> $HOME/.bashrc \
    && echo "export LANG=C.UTF-8" >> $HOME/.bashrc \
    && echo "export PYTHONPATH=$HOME/workspace/chunkflow:\$PYTHONPATH" >> $HOME/.bashrc  

WORKDIR $HOME/workspace/chunkflow/bin
