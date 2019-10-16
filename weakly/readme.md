# Weakly-supervised registration tutorial

Yipeng Hu

## Tutorials
There are two parts.
### Training 
on [Notebook][notebook_training]
### Inference 
on [Notebook][notebook_inference]
### The porject 
on [Azure][azure]

[azure]: https://notebooks.azure.com/yipeng-hu/projects/learn2reg-tutorials-weakly

[notebook_training]: ./tutorial_training.ipynb
[notebook_inference]: ./tutorial_inference.ipynb


## Instructions for Anaconda enviroment settings:
### 1 - Install Anaconda
[Download][anaconda_install]

[anaconda_install]:https://www.anaconda.com/distribution/

### 2 - Install tensorflow
In Anaconda Prompt, type:
```
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env
```
or 
```
conda create -n tensorflow_gpuenv tensorflow-gpu
conda activate tensorflow_gpuenv
```

### 3 - Install nibabel
conda install -c conda-forge nibabel


### 4 - Install notebook in tensorflow_env
Use the Anaconda Navigator: <\br>
Change the enviroment to tensorflow_env (or tensorflow_gpuenv); <\br>
Click to install Jupyter Notebook.


### 5 - Open the notebook
In Anaconda Navigator, click to launch the notebook. <\br>
or <\br>
In Anaconda Prompt, type:
```
conda activate tensorflow_env
jupyter notebook
```