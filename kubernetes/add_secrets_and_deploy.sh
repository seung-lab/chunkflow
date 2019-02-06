#!/bin/sh
# connect to cluster
gcloud container clusters get-credentials $1 --region us-central1 --project iarpa-microns-seunglab
#install NVIDIA GPU device drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
#create secrets for AWS and Google cloud
kubectl create secret generic secrets --from-file=$HOME/.cloudvolume/secrets/google-secret.json --from-file=$HOME/.cloudvolume/secrets/microns-seunglab-google-secret.json --from-file=$HOME/.cloudvolume/secrets/aws-secret.json
kubectl apply -f $2 
