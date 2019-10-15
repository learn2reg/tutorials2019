# Weakly-supervised registration tutorial

Yipeng Hu

## Tutorials
There are two parts.
### Training 
[Notebook][notebook_training]
[Kaggle][kaggle_train]
### Inference
[Notebook][notebook_inference]
[Kaggle][kaggle_inference]

[notebook_training]: ./tutorial_training.ipynb
[notebook_inference]: ./tutorial_inference.ipynb

[kaggle_train]: ./tutorial_training.ipynb
[kaggle_inference]: ./tutorial_inference.ipynb


## Instructions for Kaggle enviroment settings:
- to be validated


## Instructions for Anaconda enviroment settings:
### 1 - Install Anaconda
[Download][anaconda_install]

[anaconda_install]:https://www.anaconda.com/distribution/

### 2 - Install tensorflow
In Anaconda Prompt, type:
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env

or 

conda create -n tensorflow_gpuenv tensorflow-gpu
conda activate tensorflow_gpuenv

### 3 - Install nibabel
conda install -c conda-forge nibabel


### 4 - Install notebook in tensorflow_env
Use the Anaconda Navigator:
Change the enviroment to tensorflow_env (or tensorflow_gpuenv)
Click to install Jupyter Notebook


### 5 - Open the notebook
In Anaconda Navigator, click to launch the notebook

or 

In Anaconda Prompt, type:
conda activate tensorflow_env
jupyter notebook