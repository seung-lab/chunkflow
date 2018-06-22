ARG BACKEND=pznet

FROM jingpengw/chunkflow:backend-${BACKEND}


LABEL maintainer = "Jingpeng Wu" \
    email = "jingpeng@princeton.edu"

WORKDIR /root 
RUN mkdir chunkflow
ADD . chunkflow/ 

WORKDIR ./chunkflow 

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install -r requirements.txt 

ENV PYTHONPATH /root/chunkflow/python:$PYTHONPATH 
