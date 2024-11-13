```
                  .                                                         
               ,'/ \`.                                                               
              |\/___\/|                                                     
              \'\   /`/          ██╗ █████╗ ███████╗ ██████╗ ███████╗██████╗
               `.\ /,'           ██║██╔══██╗██╔════╝██╔════╝ ██╔════╝██╔══██╗                   
                  |              ██║███████║█████╗  ██║  ███╗█████╗  ██████╔╝ 
                  |         ██   ██║██╔══██║██╔══╝  ██║   ██║██╔══╝  ██╔══██╗
                 |=|        ╚█████╔╝██║  ██║███████╗╚██████╔╝███████╗██║  ██║
            /\  ,|=|.  /\    ╚════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝
        ,'`.  \/ |=| \/  ,'`.                                                 
      ,'    `.|\ `-' /|,'    `.                                              
    ,'   .-._ \ `---' / _,-.   `.                                            
       ,'    `-`-._,-'-'   `.       
      '  
```



Jaeger : an accurate and fast deep-learning tool to detect bacteriophage sequences
===============
![GitHub](https://img.shields.io/github/license/Yasas1994/jaeger) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Yasas1994/jaeger/main?color=8a35da) ![Conda](https://img.shields.io/conda/v/bioconda/jaeger-bio) ![Conda](https://img.shields.io/conda/dn/bioconda/jaeger-bio) [![PyPI version](https://badge.fury.io/py/jaeger-bio.svg)](https://badge.fury.io/py/jaeger-bio) [![Downloads](https://static.pepy.tech/badge/jaeger-bio)](https://pepy.tech/project/jaeger-bio) [![DOI](https://zenodo.org/badge/379281156.svg)](https://zenodo.org/doi/10.5281/zenodo.13336194)



Jaeger is a tool that utilizes homology-free machine learning to identify phage genome sequences that are hidden within metagenomes. It is capable of detecting both phages and prophages within metagenomic assemblies.

---
#### Citing Jaeger
---

If you use Jaeger in your work, please consider citing its preprint: 

* <b>Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences</b>
Yasas Wijesekara, Ling-Yi Wu, Rick Beeloo, Piotr Rozwalak, Ernestina Hauptfeld, Swapnil P. Doijad, Bas E. Dutilh, Lars Kaderali bioRxiv 2024.09.24.612722

To cite the code itself:

* <b>Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences</b> [![DOI](https://zenodo.org/badge/379281156.svg)](https://zenodo.org/doi/10.5281/zenodo.13336194)

---



- [Installing Jaeger](#installation)
  - [Bioconda](#option-1--bioconda)
  - [PyPi](#option-2--installing-from-pypi)
  - [git (dev-version)](#option-3--installing-from-git)
- [Troubleshooting](#troubleshooting)
- [Running Jaeger](#running-jaeger)
- [What is in the output](#what-is-in-the-output)
- [Predicting prophages](#predicting-prophages-with-jaeger)

--- 
#### Installing Jaeger
---

##### option 1 : bioconda

The performance of the Jaeger workflow can be significantly increased by utilizing GPUs. To enable GPU support, the CUDA Toolkit and cuDNN library must be accessible to conda.

````
# setup bioconda
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict

# create conda environment and install jaeger
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.9,<3.12" pip jaeger-bio


# activate environment
conda activate jaeger
````
Test the installation with test data
```
jaeger test
```


##### option 2 : Installing from pypi



```
# create a conda environment and activate  
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.9,<3.12" pip
conda activate jaeger

# OR create a virtual environment using venv
python3 -m venv jaeger
source jaeger/bin/activate    

# to install jaeger with GPU support
pip install jaeger-bio[gpu]

# to install without GPU support
pip install jaeger-bio[cpu]

# to install on a Mac(arm)
pip install jaeger-bio[darwin-arm]


```

##### option 3 : Installing from git

```
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


---
##### Troubleshooting
---

If you have a NVIDIA GPU on the system, and jaeger fails to detect it, try these steps.

1. If you are on a HPC check whether cuda-toolkit is available as a module. (Skip this step if you are trying this out on your PC)

```
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
```
module load cuda/12.0.0
```

2. Next, check whether the [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) is properly configured.

````
nvidia-smi
````

Above command returns the following output if everything is properly set-up. You can also determine the cuda version from it. For example here it is 11.7 (for step 3)
````
Mon Apr  8 14:26:43 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1660 Ti     Off | 00000000:01:00.0 Off |                  N/A |
| N/A   47C    P8               2W /  80W |      6MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2196      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
````

Check whether Jaeger detects the GPU now.

If that fails you will have to manually configure the conda environment as shown in step 3.

3. * cuda-toolkit for cuda>=11.1 can be found here https://anaconda.org/nvidia/cuda-toolkit 

Following example shows the installation process for cuda=11.3.0. Simply change the version number on the second "nvidia/label/cuda-11.x.x" command to install a different version
 
````
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

---
#### Running Jaeger
---
##### CPU/GPU mode
Once the environment is properly set up, using Jaeger is straightforward. The program can accept both compressed and uncompressed .fasta files containing the contigs as input. It will output a table containing the predictions and various statistics calculated during runtime. 

```
jaeger run -i input_file.fasta -o output_dir --batch 128
```
##### multi-GPU mode

We provide a new program that allows users to automatically run multiple instances of Jaeger on several GPUs allowing maximum utilization of state-of-the-art hardware. This program accepts a file with a list of paths to all input FASTA files. **--ngpu** flag can be used to set the number of GPUs at your disposal. **--maxworkers** flag can be used to set the number of samples that should be processed parallaly per GPU. All other arguments remains similar to 'Jaeger' program.


```
# to generate a list of fasta files in a dir
ls ./files/*.fna | xargs realpath > input_file_list

# to process eight samples in parallel on two GPUs 
jaeger_parallel -i input_file_list -o output_dir --batch 128 --maxworkers 4 --ngpu 2
```

##### Selecting the batch parameter 

You can control the number of parallel computations using this parameter. By default it is set to 96. If you run into OOM errors, please consider setting the --bactch option to a lower value. for example 96 is good enough for a graphics card with 4 Gb of memory.

---
#### What is in the output?
---
All predictions are summarized in a table located at ```output_dir/<input_file>_default.jaeger.tsv```

```
┌───────────────────────────────────┬────────┬────────────┬─────────┬───┬─────────────┬────────────────┬──────────────────┬───────────────┐
│ contig_id                         ┆ length ┆ prediction ┆ entropy ┆ … ┆ Archaea_var ┆ window_summary ┆ terminal_repeats ┆ repeat_length │
╞═══════════════════════════════════╪════════╪════════════╪═════════╪═══╪═════════════╪════════════════╪══════════════════╪═══════════════╡
│ NODE_1109_length_9622_cov_23.163… ┆ 9622   ┆ Phage      ┆ 0.43    ┆ … ┆ 0.143       ┆ 1V1n2V         ┆ null             ┆ null          │
│ NODE_1181_length_9275_cov_26.864… ┆ 9275   ┆ Phage      ┆ 0.327   ┆ … ┆ 0.504       ┆ 4V             ┆ null             ┆ null          │
│ NODE_123_length_36569_cov_24.228… ┆ 36569  ┆ Phage      ┆ 0.503   ┆ … ┆ 1.554       ┆ 9V1n7V         ┆ null             ┆ null          │
│ NODE_149_length_32942_cov_23.754… ┆ 32942  ┆ Phage      ┆ 0.458   ┆ … ┆ 3.229       ┆ 3V1n1n11V      ┆ null             ┆ null          │
│ NODE_231_length_24276_cov_21.832… ┆ 24276  ┆ Phage      ┆ 0.502   ┆ … ┆ 1.467       ┆ 1V1n3V1n5V     ┆ null             ┆ null          │
└───────────────────────────────────┴────────┴────────────┴─────────┴───┴─────────────┴────────────────┴──────────────────┴───────────────┘

```

This table provides information about various contigs in a metagenomic assembly. Each row represents a single contig, and the columns provide information about the contig's ID, length, the number of windows identified as prokaryotic, viral, eukaryotic, and archaeal, the prediction of the contig (Phage or Non-phage), the score of the contig for each category (bacterial, viral, eukaryotic and archaeal), and a summary of the windows. The table can be used to identify potential phage sequences in the metagenomic assembly based on the prediction column. The score columns can be used to further evaluate the confidence of the prediction and the window summary column can be used to understand the count of windows that contributed to the final prediction.

---

#### Options
---

````
jaeger run --help
````
````

## Jaeger 1.1.30 (yet AnothEr phaGe idEntifier) Deep-learning based bacteriophage discovery 
https://github.com/Yasas1994/Jaeger.git
usage: jaeger run  -i INPUT -o OUTPUT

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input file
  -o OUTPUT, --output OUTPUT
                        path to output directory
  --fsize [FSIZE]       length of the sliding window (value must be 2^n). default:2048
  --stride [STRIDE]     stride of the sliding window. default:2048 (stride==fsize)
  -m {default,experimental_1,experimental_2}, --model {default,experimental_1,experimental_2}
                        select a deep-learning model to use. default:default
  -p, --prophage        extract and report prophage-like regions. default:False
  -s [SENSITIVITY], --sensitivity [SENSITIVITY]
                        sensitivity of the prophage extraction algorithm (between 0 - 4). default: 1.5
  --lc [LC]             minimum contig length to run prophage extraction algorithm. default: 500000 bp
  --rc [RC]             minium reliability score required to accept predictions. default: 0.2
  --pc [PC]             minium phage score required to accept predictions. default: 3
  --batch [BATCH]       parallel batch size, set to a lower value if your gpu runs out of memory. default:96
  --workers [WORKERS]   number of threads to use. default:4
  --getalllogits        writes window-wise scores to a .npy file
  --getsequences        writes the putative phage sequences to a .fasta file
  --cpu                 ignore available gpus and explicitly run jaeger on cpu. default: False
  --physicalid [PHYSICALID]
                        sets the default gpu device id (for multi-gpu systems). default: 0
  --getalllabels        get predicted labels for Non-Viral contigs. default: False
  -v, --verbose         Verbosity level : -vvv warning, -vv info, -v debug, (default info)

Misc. Options:
  -f, --overwrite       Overwrite existing files


````
---
#### Python Library
---
Jaeger can be integrated into python scripts using the jaegeraa python library as follows.
currently the predict function accepts 4 different input types.
1) Nucleotide sequence -> str
2) List of Nucleotide sequences -> list(str,str,..)
3) python file object -> (io.TextIOWrapper)
4) python generator object that yields Nucleotide sequences as str (types.GeneratorType)
5) Biopython Seq object

```python
from jaegeraa.api import Predictions

model=Predictor()
predictions=model.predict(input,stride=2048,fragsize=2048,batch=100)
model.predict()

```

returns a dictionary of lists in the following format

```python
{'contig_id': ['seq_0', 'seq_1'],
 'length': [19000, 10503],
 '#num_prok_windows': [0, 0],
 '#num_vir_windows': [9, 0],
 '#num_fun_windows': [0, 5],
 '#num_arch_windows': [0, 0],
 'prediction': ['Phage', 'Non-phage'],
 'bac_score': [-1.9552012549506292, -1.9441368103027343],
 'vir_score': [6.6312947273254395, -3.097817325592041],
 'fun_score': [-5.712721400790745, -0.6870137214660644],
 'arch_score': [-2.4369852013058133, -0.8941479325294495],
 'window_summary': ['9V', '5n']}
 
```
This dictionary can be easily converted to a pandas dataframe using DataFrame.from_dict() method
```python
import pandas as pd
df = DataFrame.from_dict(predictions)
```
---
#### Notes
---
* The program expects the input file to be in .fasta format.
* The program uses a sliding window approach to scan the input sequences, so the stride argument determines how far the window will move after each scan.
* The batch argument determines how many sequences will be processed in parallel.
* The program is compatible with both CPU and GPU. By default, it will run on the GPU, but if the --cpu option is provided, it will use the specified number of threads for inference.
* The program uses a pre-trained neural network model for phage genome prediction.
* The --getalllabels option will output predicted labels for Non-Viral contigs, which can be useful for further analysis.
It's recommended to use the output of this program in conjunction with other methods for phage genome identification.

---

#### Predicting prophages with Jaeger
---

```
jaeger run -p -i NC_002695.fna -o outdir 
```
The outdir will contain the following files
```
|____Escherichia_coli_O157-H7_prophages
| |____plots
| | |____NC_002695_Escherichia_coli_O157-H7_jaeger.pdf
| |____prophages_jaeger.tsv
|____Escherichia_coli_O157-H7_jaeger.log
|____Escherichia_coli_O157-H7_default_jaeger.tsv
```

users can find the following visulaization in the ```plots``` directory <br><br>

<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/3efcd886-e45a-454f-9f61-53f954932b84"  width="500">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Yasas1994/Jaeger/assets/34155351/6acc1561-2c36-42c5-94ba-523721e902a5"  width="500">
  <img alt="dark mode" src="https://github.com/Yasas1994/Jaeger/assets/34155351/3efcd886-e45a-454f-9f61-53f954932b84">
</picture>
</p>

 <br><br>


list of prophage coordinates can be found in ```prophages_jaeger.tsv```
```
┌─────────────┬────────────┬──────────┬──────────┬───┬──────────┬────────┬────────────┬────────────┐
│ contig_id   ┆ alignment_ ┆ identiti ┆ identity ┆ … ┆ gc%      ┆ reject ┆ attL       ┆ attR       │
│             ┆ length     ┆ es       ┆          ┆   ┆          ┆        ┆            ┆            │
╞═════════════╪════════════╪══════════╪══════════╪═══╪══════════╪════════╪════════════╪════════════╡
│ NC_002695   ┆ 16.0       ┆ 16.0     ┆ 1.0      ┆ … ┆ 0.435049 ┆ false  ┆ GCACCATTTA ┆ GCACCATTTA │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ AATCAA     ┆ AATCAA     │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 15.0       ┆ 15.0     ┆ 1.0      ┆ … ┆ 0.493497 ┆ false  ┆ GCTTTTTTAT ┆ GCTTTTTTAT │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ ACTAA      ┆ ACTAA      │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 60.0       ┆ 60.0     ┆ 1.0      ┆ … ┆ 0.511819 ┆ false  ┆ TGGCGGAAGC ┆ TGGCGGAAGC │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ GCAGAGATTC ┆ GCAGAGATTC │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆ GAACTCTGGA ┆ GAACTCTGGA │
│             ┆            ┆          ┆          ┆   ┆          ┆        ┆ AC…        ┆ AC…        │
│ NC_002695   ┆ 16.0       ┆ 16.0     ┆ 1.0      ┆ … ┆ 0.499516 ┆ false  ┆ TTCTTTATTA ┆ TTCTTTATTA │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ CCGGCG     ┆ CCGGCG     │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
│ NC_002695   ┆ 14.0       ┆ 14.0     ┆ 1.0      ┆ … ┆ 0.529465 ┆ false  ┆ CGTCATCAAG ┆ CGTCATCAAG │
│ Escherichia ┆            ┆          ┆          ┆   ┆          ┆        ┆ TGCA       ┆ TGCA       │
│ coli O157-… ┆            ┆          ┆          ┆   ┆          ┆        ┆            ┆            │
└─────────────┴────────────┴──────────┴──────────┴───┴──────────┴────────┴────────────┴────────────┘

```
---
#### Visualizing predictions 
---

You can use [phage_contig_annotator](https://github.com/Yasas1994/phage_contig_annotator) to annotate and visualize Jaeger predictions.

---
#### Acknowlegements
---

This work was supported by the European Union’s Horizon 2020 research and innovation program, under the Marie Skłodowska-Curie Actions Innovative Training Networks grant agreement no. 955974 ([VIROINF](https://viroinf.eu/)), the European Research Council (ERC) Consolidator grant 865694 

<img src=https://github.com/Yasas1994/Jaeger/assets/34155351/0cad76c6-6e4d-4b89-8e73-257137cf73a8 width="210" height="84">  &nbsp;&nbsp;&nbsp;  <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/fef3bc35-8a8c-44c9-85ca-35ab0c68130e width="100" height="100">  &nbsp;&nbsp;&nbsp;   <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/f15ab9b6-cade-4315-941c-e897f753dad9 width="150" height="100">

The ascii art logo is from  <font size="3"> https://ascii.co.uk/ </font>


