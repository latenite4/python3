#!/bin/bash
#gcloud script to create GCE TPU instance
# name: R. Melton
# date: 1/3/2021

set -x

GCP_ZONE=us-central1-c
TF_VERSION=2.4.0

#ctpu up --project=${GCP_PROJECT} --zone=us-central1-b --tf-version=2.4.0 --name=first-tpu
#gcloud alpha compute tpus create

gcloud alpha compute tpus create --zone=$GCP_ZONE --version=$TF_VERSION --description="NIST classification"  tpu-nist
gcloud alpha compute tpus list