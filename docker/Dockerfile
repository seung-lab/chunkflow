# VERSION 0.0.5-customAF
# AUTHOR: Will Wong
# DESCRIPTION: Alpine base image with dockerized airflow and ECR registry and DooD (Docker outside of Docker)
# BUILD: docker build --rm -t wongwill86/air-tasks:base-alpine -f /docker/base/Dockerfile.base-slim .
# SOURCE: https://github.com/wongwill86/air-tasks

FROM python:3.6-alpine3.7

ENV CHUNKFLOW_USER=chunkflow
ENV CHUNKFLOW_HOME=/usr/local/chunkflow

WORKDIR ${CHUNKFLOW_HOME}
RUN set -ex \
    && apk add --no-cache bash \
    && apk add --no-cache --virtual .build-dependencies \
        build-base \
        python-dev \
        libffi-dev \
        openblas-dev \
        linux-headers \
        jpeg-dev \
        zlib-dev \
        shadow \
    && pip install --upgrade pip \
    && groupadd -r ${CHUNKFLOW_USER} \
    && useradd -r -d ${CHUNKFLOW_HOME} -g ${CHUNKFLOW_USER} -s /bin/bash ${CHUNKFLOW_USER} \
    && chown -R ${CHUNKFLOW_USER}: ${CHUNKFLOW_HOME} \
    && find /usr/local \
        \( -type d -a -name test -o -name tests \) \
        -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
        -exec rm -rf '{}' + \
    && runDeps="$( \
        scanelf --needed --nobanner --recursive /usr/local \
                | awk '{ gsub(/,/, "\nso:", $2); print "so:" $2 }' \
                | sort -u \
                | xargs -r apk info --installed \
                | sort -u \
    )" \
    && apk add --virtual .run-dependencies $runDeps

COPY chunkflow/ .
COPY setup.py/ ./
COPY data/ .
RUN pip install . && pip install http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl



