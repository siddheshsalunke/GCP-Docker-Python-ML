# Deploying Machine Learning on Cloud


The Machine Learning code provided:
SampleImages_classify.py
Pretrained modules:
saved_svm_model
saved_sc

To check the code:
```
$python SampleImages_classify.py
```
Google CLI deployment of Compute Engine

```
$ gcloud compute firewall-rules create default-allow-http --allow=tcp:80 --target-tags http-server
$ gcloud compute instances create cloud-ml --machine-type=n1-standard-1 --zone=us-east1-c --tags=http-server
$ gcloud compute ssh cloud-ml --zone=us-east1-c
```

Building Image: (files mentioned are attached)
Execute:  Bring in the Libraries
```
$pip install -r requirements.txt 
```
Deployment model
- /home/Cloud-ml/
  - Dockerfile
  - nginx.conf
  - requirements.txt
  - service
    - Sampleimages_classify.py
    - saved_svm_model
    - saved_sc
  - supervisord.conf


Docker Building 
```
$docker build –t cloud-ml .
$docker tag cloud-ml us.gcr.io/cloud-ml-model/cloud-ml
$gcloud docker – push us.gcr.io/cloud-ml-model/cloud-ml
```

Run the riots ;) Using newly Docker Image
```
$ docker run -td -p 80:80 gcr.io/cloud-ml-model/cloud-ml
```
