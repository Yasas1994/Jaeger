![image](https://user-images.githubusercontent.com/34155351/162472470-1ea60946-5013-44a7-b952-31506a399f7e.png)
# Jaeger (Just Another phaGE identifieR)
Identifying phage genome sequences conclealed in metagenomes is a long standing problem in viral metagenomics and ecology. Although various methods have been introduced in the recent past, each methods has its own strengths and weaknesses. Recently, machine learning has gained popularity in detecting viruses in metagenome samples. Here, we introduce Jaeger a CNN based, sensitive deep learning method that outperforms other state-of-the-art methods.


## Installation
Jaeger is currently tested only on python 3.9.2 therefore, we recomend you to setup a conda environmet by running


````
conda create -n jaeger python=3.9.2

````

Jaeger workflow can be greatly accelarated on system with gpus.(you can expect the runtime to decrease linearly with increasing the number of gpus) To add support for gpus, cudatoolkit and cudnn has to installed on the created conda environmnet by typing (you can skip this step if you don't wish to use a gpu) 


````
conda install -n jaeger -c anaconda cudatoolkit 

conda install -n jaeger -c conda-forge cudnn

````

Then install tensorflow, numpy, tqdm on the conda environment using pip


````
pip install tensorflow==2.5 numpy==1.19.5 tqdm==4.64.0 biopython==1.78


````



