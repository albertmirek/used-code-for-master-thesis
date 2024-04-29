# ML Pipeline for MLOps Implementation

This repository contains the code used for the machine learning pipeline developed as part of the master thesis titled "Implementation of MLOps within an E-commerce Company". The codebase includes scripts for model training, data preprocessing, and other automation scripts that form the backbone of the MLOps pipeline designed for this project.

## Project Overview

The goal of this project was to integrate machine learning operations (MLOps) within the existing DevOps practices of an e-commerce company to streamline the lifecycle management of ML models.
This repository includes the code for the Neural Sequence Aware recommender as well as the code for preparing the dataset.


The whole construction of the pipeline is described in the thesis, and is not integrated in this repository, for the reasons that
GitLab CI/CD was used for the original project. However the rest of the code that was necessary for the model training and execution is present.



### Code


#### Dockerfiles
The [mlflow.Dockerfile](mlflow.Dockerfile) specifies the image for the MLflow, which is used for tracking the model experiemtns

[torchserve.Dockerfile](torchserve.Dockerfile) represents the multi stage dockerfile, where the **runtime** + **dev** targets are
used for the model training and **prod** target is used for constructing the TorchServe with packaged model


[docker-compose.yml](docker-compose.yml) includes the local orchestration components of MLflow, MySQl, S3 (Minio) and Adminer
for local tracking with Docker containers

### Pipeline
The GitLab pipeline was not implemented via GitHub Actions. The pipeline is described within thesis, however the sequential tasks
were executed via the actions located inside the [Makefile](Makefile). The actions include the training, archiving, testing adn building of the docker containers


### Training
Inside the ``/src`` directory there files used for execution of scripts for pre-processing, training and testing.

[fetch_and_prepare_raw_data.py](src/fetch_and_prepare_raw_data.py) is responsible for fetching the data from the feature store
and preparing the raw data for future modeling, where the data is firstly encoded, and scaled.


[handler.py](src/handler.py) Defines the handler which is used in the TorchServe to control the inference on the model.


[main.py](src/main.py) is the main script, which is called in the pipeline right after the raw data was prepared.
It prepares the data for model training, such as scaling and encoding the feature values. After preparing the data it instantiates
the Dataset and DataLoader functions and starts the model training and evaluation

#### Modules [``/src/modules``](src/modules) 

contains the code for the [Config.py](src/modules/Config.py) class, which specifies features and variables for preparing the dataset
toggling features, used hyperparameters for the model training, etc.

[Dataset.py](src/modules/Dataset.py) contains the class for custom PyTorch Dataset, which is invoked by the DataLoader during training
for loading the data

[Model.py](src/modules/Model.py) defines the Neural Sequence Aware recommender architecture and functionality

#### Tests
[``/src/test``](src/test/torch_serve_test.py) contains the script executed for inferring the started TorchServe container


#### Utils [``/src/utils``](src/utils) 
This directory contains the scripts for processing raw data, processing the data before modeling and train/test splitting

[pre_process_raw_data](src/utils/pre_process_raw_data.py) includes the code for transformation and filtering rules for the raw data

[prepare_for_modeling](src/utils/prepare_for_modeling.py) includes the code for scaling encoding dataset for model experimentations

[train_test_split](src/utils/train_test_split.py) includes the code used for splitting the final dataset to training and testing

