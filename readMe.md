Here we share the code for 

Task-driven neural network models predict neural dynamics of proprioception
by Alessandro Marin Vargas*, Axel Bisi*, Alberto Chiappa, Chris Versteeg, Lee Miller,  Alexander Mathis

It builds on ...


# Contrasting action and posture coding with hierarchical deep neural network models of proprioception

This repository contains code for the manuscript ["Contrasting action and posture coding with hierarchical deep neural network models of proprioception"](https://elifesciences.org/articles/81499), by Kai J Sandbrink*, Pranav Mamidanna*, Claudio Michaelis, Matthias Bethge, Mackenzie W Mathis** and Alexander Mathis** eLife 2023. A preprint was available since [2020](https://www.biorxiv.org/content/10.1101/2020.05.06.081372v1).

Highlights of the paper:

- We provide a normative approach to derive neural tuning of proprioceptive features from behaviorally-defined objectives.
- We propose a method for creating a scalable muscle spindles dataset based on kinematic data and define an action recognition task as a benchmark.
- Hierarchical neural networks solve the recognition task from muscle spindle inputs.
- Individual neural network units resemble neurons in primate somatosensory cortex, and networks make predictions for other areas along the proprioceptive pathway.

# Structure of the code

The code is organized as follows:

1. Dataset Generation. (Proprioceptive Character Recognition Task = PCRT Dataset) Code can be found in `dataset`
2. Classifying the PCRT Dataset using binary SVMs. Code can be found in `svm-analysis`
3. Classifying the PCRT Dataset using various network models. Code in `nn-training`
4. Representational Similarity (RDM) analysis of the models. Code in `repr-analysis`
5. Single Unit tuning curve, tSNE, CKA analysis. Code in `single_cell`

`code` contains classes and helper functions used at one/more places in the analyses.

Each part of the code has a dedicated readMe.md to describe how to run this section (e.g. `/single_cell/readMe.md`).

# Runtimes and datasets:

Installation should take a few minutes (for the conda environment) and half an hour for the docker container (see below). All runtimes are for a strong computer (CPU) except the NN training, which is for a GPU.

1. Dataset generation: one run of generate_dataset.py (generates up to 200 datapoints) = 1 hour.
2. SVM analysis: train one pairwise SVM using binary_svm.py on all datasets (full dataset, augmentation suppressed) = 2-4 hours.
3. NN Training: train one neural network using train_convnets.py = 2-4 hours. (on a GPU)
4. RDM: generate responses and compute rdms for all layers of one network on a test set of 4000 characters = 1-2 hours.
5. Extracting hidden layer activations : a few minutes (per model)
6. Fitting tuning curves for all models and planes: ~24h
7. Remaining analyses of tuning strengths and of plotting : ~3 hrs

Thus, for instance creating the PCRT dataset with 1,000,000 trajectories takes a substantial amount of time. For this reason we also share the datasets to allow reproduction of results from various points. All data are available at:
https://www.dropbox.com/sh/afvyg524bakyo4u/AADqeN0uHYFdqfEEGShQeHopa?dl=0

We share the PCRT dataset (contained in 'pcr_data') it has approximately ~30GB.

We share the weights of all the trained networks (contained in 'network-weights'): about ~3.5GB

We share the data for analysis (activations, etc. contained in 'analysis-data'): about ~88GB.

# Installation, software & requirements

This project comrpises multiple parts, which is why there are also two different computing environments for reproducing the results. Below, provide a docker container for creating the PCR dataset and training the models and a conda enviroment for analyzing the models. These environments are not cross-compatible. 

## Creating the dataset/training DNNs (via docker)

Dataset generation requires [OpenSim](https://opensim.stanford.edu/) and the network training requires [TensorFlow](https://www.tensorflow.org/). To easily reproduce our computational environment incl. the dependencies we are sharing a Docker container with OpenSim binaries and TensorFlow. It is available here: https://hub.docker.com/r/pranavm19/opensim/tags

Starting the docker container from the image. After pulling the docker image from the docker hub, in the terminal, start the container with the following command:

Options:

* change port: (i.e. 2355 can be 777, etc)
* change which GPU to use (check which GPU you want to use in the terminal by running nvidia-smi)
* change the name: --name containername can be anything you want

```
GPU=0 bash ./dlc-docker run -d -p 2355:8888 -v $MEDIA --name containername pranavm19/opensim:opensim-tf
```

Enter the container via the terminal (to get terminal access in container):

```
docker exec --user $USER -it containername /bin/bash
```

In the container, check that OpenSim is correctly imported:

```
python3.6
import opensim
```

After finishing, the container can be stopped:

```
docker stop containername
```

To start the container again:

```
docker start containername
```

### Jupyter and docker

You can access Jupyter by going to the port you specified (e.g. http://localhost:2355) in Google Chrome. 

To get the token for entry, back in the terminal, look at the docker log:

```
docker logs containername 
```

Copy and paste the value after "token=". 


## Reproducing the analysis (after dataset creation) 

For the rest of the analysis, we are sharing a *conda environment that has the dependencies*. It can be installed by:

```
conda env create -f environment.yml
source activate DeepDraw
```

It was exported on an Ubuntu system, where all the analyses were performed.
```
conda env export > environment.yml
```
### References

This repository contains code for the manuscript: Kai J Sandbrink, Pranav Mamidanna, Claudio Michaelis, Matthias Bethge, Mackenzie Weygandt Mathis, Alexander Mathis (2023) Contrasting action and posture coding with hierarchical deep neural network models of proprioception eLife 12:e81499

Paper: https://elifesciences.org/articles/81499


Preprint: https://www.biorxiv.org/content/10.1101/2020.05.06.081372v3



