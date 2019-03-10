# Usage of kubernetes

## [Install NVIDIA drivers](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)
```
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## build secret pod for mounting
```
kubectl create secret generic secrets \
--from-file=/secrets/google-secret.json \
--from-file=/secrets/aws-secret.json \
--from-file=/secrets/boss-secret.json
```

## reconfig cluster
```
gcloud container clusters resize my-cluster --region us-central1 --node-pool gpu-pool-1 --size 1

kubectl edit configmap kube-dns-autoscaler --namespace=kube-system
```
## reconnect
`gcloud container clusters get-credentials my-cluster`

## monitor and watch
get the pod id
    kubectl get pods

watch the logs
    watch kubectl logs pod-id

We can monitor the traffic operations per second in Kubernetes deployment monitor interface.
There is no bandwidth usage in Kubernetes, we can only monitor it in `instance group` monitor interface.

## deployment
- create: `kubectl apply -f deploy.yml`
- check:  `kubectl get deployments`
- delete: `kubectl delete deployment inference`
- sclae: `kubectl scale --replicas=85 -f deploy.yml`
