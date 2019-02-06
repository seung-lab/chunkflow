FROM seunglab/chunkflow:latest

COPY pytorch-emvision /root/chunkflow/kubernetes/pytorch-emvision/ 
COPY pytorch-model /root/chunkflow/kubernetes/pytorch-model/

RUN echo "export PYTHONPATH=/root/chunkflow/kubernetes/pytorch-model:/root/chunkflow/kubernetes/pytorch-emvision:\$PYTHONPATH" >> /root/.bashrc;

WORKDIR /root/chunkflow/scripts

