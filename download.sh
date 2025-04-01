#!/bin/bash
mkdir data/
curl -L -o data/mnist-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
cd data && unzip mnist-dataset.zip