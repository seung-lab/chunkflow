FROM seunglab/kaffe:pznet

LABEL maintainer="Jingpeng Wu"\
    email="jingpeng@princeton.edu"

RUN apt-get update && apt-get install -y python3-pip
ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:$PYTHONPATH
