#### option 1 : bioconda

The performance of the Jaeger workflow can be significantly increased by utilizing GPUs. To enable GPU support, the CUDA Toolkit and cuDNN library must be accessible to conda.

````bash
# setup bioconda
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
````
create conda environment and install jaeger
```bash
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.9,<3.12" pip jaeger-bio
```

activate the environment
```bash
conda activate jaeger
```

test the installation with test data
```bash
jaeger test
```


#### option 2 : Installing from pypi


create a conda environment and activate
```bash
  
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.9,<3.12" pip
conda activate jaeger
```
OR create a virtual environment using venv
```bash
python3 -m venv jaeger
source jaeger/bin/activate    

```
to install jaeger with GPU support
```bash
pip install jaeger-bio[gpu]
```
to install without GPU support
```bash
pip install jaeger-bio[cpu]
```

to install on a Mac(arm)
```bash
pip install jaeger-bio[darwin-arm]
```

#### option 3 : Installing from git

```bash
# clone the jaeger repository
git clone https://github.com/MGXlab/Jaeger.git
cd Jaeger

# create a conda environment and activate  
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.9,<3.12" pip
conda activate jaeger

# OR create a virtual environment using venv
python3 -m venv jaeger
source jaeger/bin/activate    

# install jaeger

# to install with GPU support
pip install ".[gpu]"

# to install without GPU support
pip install ".[cpu]"

# to install on a Mac(arm)
pip install ".[darwin-arm]"

```

#### option 4 : Apptainer (recomended)
```bash
# clone the jaeger repository
git clone https://github.com/MGXlab/Jaeger.git
cd Jaeger

# to build the container
apptainer build jaeger.sif singularity/jaeger_singularity.def

# to run jaeger
apptainer run --nv jaeger.sif jaeger run -i input_file.fasta -o output_dir --batch 128
```


#### Troubleshooting


If you have a NVIDIA GPU on the system, and jaeger fails to detect it, try these steps.

1. If you are on a HPC check whether cuda-toolkit is available as a module. (Skip this step if you are trying this out on your PC)

```bash
module avail
```

```
angsd/0.937         boost/1.71.0        clang/14.0.4  fastp/0.23.1   gcc/13.2.0     julia/1.9.2         modeller/9.23      proj/7.0.1          structure/2.3.4     vcftools/0.1.16  
autodockvina/1.1.2  boost/1.79.0        clang/17.0.5  fastqc/0.11.9  hdf5/1.12.1    kalign/1.04         mrbayes/3.2.7      r/4.1.1             superlu-dist/8.1.2  
bamutil/1.0.15      bowtie/2.4.2        colmap/3.8    fgsl/1.5.0     hdf5/1.14.0    likwid/5.2.0        openmpi/4.1.1      r/4.3.1             superlu-dist/8.2.0  
baypass/2.2         bwa/0.7.17          cuda/11.4     fsl/6.0.2      hhsuite/3.3.0  likwid/5.2.1        openpmix/3.1.5     samtools/1.12       superlu/4.3         
bcftools/1.15       cdhit/4.8.1         cuda/11.7     gams/36.2.0    I-TASSER/5.1   mathematica/13.2.1  petsc-real/3.18.1  singularity/3.10.0  transdecoder/5.7.0  
bedtools/2.30.0     ceres-solver/2.1.0  cuda/12.0.0   gcc/12.2.0 
```

If so, load it
```bash
module load cuda/12.0.0
```

2. Next, check whether the [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) is properly configured.

````bash
nvidia-smi
````

Above command returns the following output if everything is properly set-up. You can also determine the cuda version from it. For example here it is 11.7 (for step 3)
````
Fri Nov 15 07:46:02 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.216.01             Driver Version: 535.216.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-80GB          On  | 00000000:01:00.0 Off |                    0 |
| N/A   41C    P0              74W / 500W |   3616MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-SXM4-80GB          On  | 00000000:41:00.0 Off |                    0 |
| N/A   40C    P0              65W / 500W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-SXM4-80GB          On  | 00000000:81:00.0 Off |                    0 |
| N/A   39C    P0              65W / 500W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100-SXM4-80GB          On  | 00000000:C1:00.0 Off |                    0 |
| N/A   40C    P0              65W / 500W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   3583343      C   /opt/conda/bin/python                      3606MiB |
+---------------------------------------------------------------------------------------+

````

Check whether Jaeger detects the GPU now.

If that fails you will have to manually configure the conda environment as shown in step 3.

3. * cuda-toolkit for cuda>=11.1 can be found here https://anaconda.org/nvidia/cuda-toolkit 

Following example shows the installation process for cuda=11.3.0. Simply change the version number on the second "nvidia/label/cuda-11.x.x" command to install a different version
 
````bash
libcudnn_cnn_infer.so.8

# create a conda environment
conda create -n jaeger -c conda-forge -c bioconda -c defaults "python>=3.9,<3.12" pip

# cudatoolkit and cudnn
conda install -n jaeger -c "nvidia/label/cuda-11.3.0" cudatoolkit=11
conda install -n jaeger -c conda-forge cudnn

# install jaeger
conda install -n jaeger -c conda-forge -c bioconda -c defaults jaeger-bio

# activate environment
conda activate jaeger
````
More information on properly setting setting up tensorflow can be found [here](https://www.tensorflow.org/install/pip)