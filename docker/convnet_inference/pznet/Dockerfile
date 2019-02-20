FROM seunglab/kaffe:pznet

LABEL maintainer="Jingpeng Wu"\
    email="jingpeng@princeton.edu"

RUN apt-get update && apt-get install -y python-pip3
ENV PYTHONPATH /opt/znnphi_interface/code/znet/src/python:$PYTHONPATH
