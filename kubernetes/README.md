# Usage of kubernetes

## [Install NVIDIA drivers](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## build secret pod for mounting
```
kubectl create secret generic secrets \
--from-file=google-secret.json=/secrets/google-secret.json \
--from-file=aws-secret.json=/secrets/aws-secret.json \
--from-file=boss-secret.json=/secrets/boss-secret.json
```

## reconfig cluster
```
gcloud container clusters resize my-cluster --size 5 --zone us-east1-b

kubectl edit configmap kube-dns-autoscaler --namespace=kube-system
```
## reconnect
`gcloud container clusters get-credentials my-cluster`

## deployment
- create: `kubectl create -f deploy.yml --record`
- check:  `kubectl get deployments`
- delete: `kubectl delete deployment inference`
- sclae: `kubectl scale --replicas=85 -f deploy.yml`
