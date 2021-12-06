# backend: base | pytorch | pznet 
ARG BACKEND=pytorch

FROM seunglab/chunkflow:${BACKEND}
#FROM seunglab/pznet:latest

target maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

RUN mkdir -p $HOME/workspace/chunkflow

# WORKDIR only works with ENV     

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV HOME /root

WORKDIR $HOME/workspace/chunkflow
COPY . .

RUN apt-get update && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        wget \
        git \
        build-essential \
        parallel \
    # test whether pip is working 
    # there is an issue of pip:
    # https://github.com/laradock/laradock/issues/1496
	# we need this hash to solve this issue
    # && ln -sf /usr/bin/pip3 /usr/bin/pip \
    # this do not work due to an issue in pip3
    # https://github.com/pypa/pip/issues/5240
    && pip install -U pip \
    && hash -r pip \
    && pip install --upgrade setuptools \
    && pip install numpy setuptools cython --no-cache-dir \ 
    && pip install -U protobuf scipy brotlipy --no-cache-dir \
    # && pip install fpzip --no-binary :all: --no-cache-dir \
    # setup environment variables 
    # we have to setup first, otherwise click installation will fail
    && echo "export LC_ALL=C.UTF-8" >> $HOME/.bashrc \
    && echo "export LANG=C.UTF-8" >> $HOME/.bashrc \
    && echo "export PYTHONPATH=$HOME/workspace/chunkflow:\$PYTHONPATH" >> $HOME/.bashrc \
    && pip install -r requirements.txt --no-cache-dir \
    && pip install -r tests/requirements.txt --no-cache-dir \
    # install the commandline chunkflow
    && pip install -e . \
    # cleanup system libraries 
    && apt-get remove --purge -y  \
		build-essential \
    && apt-get clean \
    && apt-get autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    # the test will not pass due to missing of credentials.
    # && pytest tests \
    && chunkflow

WORKDIR $HOME/workspace/chunkflow/
