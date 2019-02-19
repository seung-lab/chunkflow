FROM seunglab/chunkflow:latest

COPY pytorch-emvision $HOME/workspace/chunkflow/kubernetes/pytorch-emvision/ 
COPY pytorch-model $HOME/workspace/chunkflow/kubernetes/pytorch-model/

RUN echo "export PYTHONPATH=$HOME/workspace/chunkflow/kubernetes/pytorch-model:$HOME/workspace/chunkflow/kubernetes/pytorch-emvision:\$PYTHONPATH" >> $HOME/.bashrc;

WORKDIR $HOME/workspace/chunkflow/scripts

