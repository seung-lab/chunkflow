# backend: base | pytorch | pznet 
ARG BACKEND=pytorch

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
    && pip install -U pip \
    # test whether pip is working 
    # there is an issue of pip:
    # https://github.com/laradock/laradock/issues/1496
	# we need this hash to solve this issue
    && hash -r pip \ 
    && pip install numpy setuptools --no-cache-dir \ 
    && pip install fpzip --no-binary :all: --no-cache-dir \
#&& git clone --single-branch --depth 1 https://github.com/seung-lab/cloud-volume.git \
#   && pip install --no-cache-dir -r $HOME/workspace/cloud-volume/requirements.txt \
    && pip install -r requirements.txt --no-cache-dir \
    # install the commandline chunkflow
    && pip install -e . \
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
