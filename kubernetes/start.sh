#!/bin/sh
# install NVIDIA GPU device drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml

#create secrets for AWS and Google cloud
kubectl create secret generic secrets \
  --from-file=google-secret.json=/home/jingpeng/.cloudvolume/secrets/google-secret.json \
  --from-file=aws-secret.json=/home/jingpeng/.cloudvolume/secrets/aws-secret.json 
#  --from-file=matrix-secret.json=/home/jingpeng/.cloudvolume/secrets/matrix-secret.json \

kubectl create -f deploy.yaml
