# Weakly-supervised registration tutorial

Yipeng Hu

## Tutorial
The complete tutorial can be run on [Azure][azure]. 

## Local version
If you would like to run locally, there are two parts, 
the Training on [Notebook][notebook_training] and 
the Inference on [Notebook][notebook_inference].


[azure]: https://notebooks.azure.com/yipeng-hu/projects/learn2reg-tutorials-weakly

[notebook_training]: ./tutorial_training.ipynb
[notebook_inference]: ./tutorial_inference.ipynb

For example, you can use a tested enviroment based on Anaconda: 
### Instructions for Anaconda enviroment settings:
#### 1 - Install Anaconda
[Download][anaconda_install]

[anaconda_install]:https://www.anaconda.com/distribution/

#### 2 - Install tensorflow
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

#### 3 - Install nibabel
```
conda install -c conda-forge nibabel
```

#### 4 - Install notebook in tensorflow_env
Use the Anaconda Navigator:  
Change the enviroment to tensorflow_env (or tensorflow_gpuenv);  
Click to install Jupyter Notebook.


#### 5 - Open the notebook
In Anaconda Navigator, click to launch the notebook. Or,  
In Anaconda Prompt, type:
```
conda activate tensorflow_env
jupyter notebook
```

## Original version
This tutorial was adapted from the demo code on [GitHub][github_demo].

[github_demo]: https://github.com/YipengHu/label-reg