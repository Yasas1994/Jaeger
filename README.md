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
              ,'    `-`-._,-'-'    `.                                                                     
             '                                                                                    
 
```


Jaeger : A quick and precise pipeline for detecting phages in sequence assemblies.
===============


Jaeger is a tool that utilizes homology-free machine learning to identify phage genome sequences that are hidden within metagenomes. It is capable of detecting both phages and prophages within metagenomic assemblies.

---
## Installation 
---
### <u> Linux and Mac (x64_86)</u>

#####  option 1 : bioconda

The performance of the Jaeger workflow can be significantly increased by utilizing GPUs. To enable GPU support, the CUDA Toolkit and cuDNN library must be accessible to conda.

````

# create conda environment and install jaeger
conda create -n jaeger -c conda-forge -c anaconda -c bioconda jaeger

# activate environment
conda activate jaeger
````

##### troubleshooting

If you have a GPU on the system, and jaeger fails to detect it, try these steps.

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
module load cuda/11.7
```

2. Next, check whether the [NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx) is properly configured.

````
nvidia-smi
````

Above command returns the following output if everything is properly set-up. You can also determine the cuda version from it. For example here it is 11.7 (for step 3)
````
Mon Apr  8 14:26:43 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.86.01    Driver Version: 515.86.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| N/A   51C    P8     6W /  N/A |   5344MiB /  6144MiB |     27%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2198      G   /usr/lib/xorg/Xorg                 69MiB |
|    0   N/A  N/A   1247272      C   ...a3/envs/jaeger/bin/python     5271MiB |
+-----------------------------------------------------------------------------+
````

Check whether Jaeger detects the GPU now.

If that fails you will have to manually configure the conda environment as shown in step 3.

3. * cuda-toolkit for cuda>=11.1 can be found here https://anaconda.org/nvidia/cuda-toolkit (not recommended)

This example shows the installation process for cuda=11.3.0. Simply change the version number on the second "nvidia/label/cuda-11.x.x" command to install a different version
 
````
libcudnn_cnn_infer.so.8

# create a conda environment
conda create -n jaeger python=3.9 pip

# cudatoolkit and cudnn
conda install -n jaeger -c "nvidia/label/cuda-11.3.0" cudatoolkit=11
conda install -n jaeger -c conda-forge cudnn

# install jaeger
conda install -n jaeger -c conda-forge -c anaconda -c bioconda jaeger

# activate environment
conda activate jaeger
````
More inoformation on properly setting setting up tensorflow can be found [here](https://www.tensorflow.org/install/pip)

##### option 2 : Installing from pypi (not recommended)



```
# create a conda environment and activate  
conda create -n jaeger python=3.9 pip
conda activate jaeger

#install jaeger
pip install jaeger-bio
```

### <u> Mac (ARM)</u>


````
  # create a conda environment
  conda create -c conda-forge -c apple -c bioconda -c defaults -n jaeger python=3.9.2 pip tensorflow=2.6 tensorflow-deps=2.6.0 numpy=1.19.5 tqdm=4.64.0 biopython=1.78

  # install tensorflow
  conda activate jaeger
  pip install tensorflow-macos
  pip install tensorflow-metal

  # install jaeger
  pip install jaeger-bio
````




---
## Running Jaeger
---
#### CPU/GPU mode
Once the environment is properly set up, using Jaeger is straightforward. The program can accept both compressed and uncompressed .fasta files containing the contigs as input. It will output a table containing the predictions and various statistics calculated during runtime. 

```
Jaeger -i input_file.fasta -o output_dir --batch 128
```
#### multi-GPU mode

We provide a new program that allows users to automatically run multiple instances of Jaeger on several GPUs allowing maximum utilization of state-of-the-art hardware. This program accepts a csv file with paths to all input .fasta files. Column with the file paths should be named as 'paths'. All other arguments remains similar to 'Jaeger' program.

``` 
Jaeger_parallel -i input_file.csv -o output_dir --batch 128 
```

#### Selecting the batch parameter 

You can control the number of parallel computations using this parameter. By default it is set to 512. If you run into OOM errors, please consider setting the --bactch option to a lower value. for example 128 is good enough for a graphics card with 6 Gb of memory.

#### options

````
Jaeger --help
````
````

## Jaeger 1.1.25 (yet AnothEr phaGe idEntifier) Deep-learning based bacteriophage discovery 
https://github.com/Yasas1994/Jaeger.git

optional arguments:
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
  --batch [BATCH]       parallel batch size, set to a lower value if your gpu runs out of memory. default:96
  --workers [WORKERS]   number of threads to use. default:4
  --getalllogits        return position-wise logits for each prediction window as a .npy file
  --usecutoffs          use cutoffs to obtain the class prediction
  --cpu                 ignore available gpus and explicitly run jaeger on cpu. default: False
  --virtualgpu          create and run jaeger on a virtualgpu. default: False
  --physicalid [PHYSICALID]
                        sets the default gpu device id (for multi-gpu systems). default:0
  --getalllabels        get predicted labels for Non-Viral contigs. default:False

Misc. Options:
  -v, --verbose         Verbosity level : -v warning, -vv info, -vvv debug, (default info)
  -f, --overwrite       Overwrite existing files
  --progressbar         show progress bar

  

````

#### Python Library
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

#### Notes
* The program expects the input file to be in .fasta format.
* The program uses a sliding window approach to scan the input sequences, so the stride argument determines how far the window will move after each scan.
* The batch argument determines how many sequences will be processed in parallel.
* The program is compatible with both CPU and GPU. By default, it will run on the GPU, but if the --cpu option is provided, it will use the specified number of threads for inference.
* The program uses a pre-trained neural network model for phage genome prediction.
* The --getalllabels option will output predicted labels for Non-Viral contigs, which can be useful for further analysis.
It's recommended to use the output of this program in conjunction with other methods for phage genome identification.

---
### What is in the output?
---

| contig_id                           |   length | prediction   |   entropy |   realiability_score | host_contam   | prophage_contam   |   #_Bacteria_windows |   #_Phage_windows |   #_Eukarya_windows |   #_Archaea_windows |   Bacteria_score |   Bacteria_var |   Phage_score |   Phage_var |   Eukarya_score |   Eukarya_var |   Archaea_score |   Archaea_var | window_summary   
|:------------------------------------|---------:|:-------------|----------:|---------------------:|:--------------|:------------------|---------------------:|------------------:|--------------------:|--------------------:|-----------------:|---------------:|--------------:|------------:|----------------:|--------------:|----------------:|--------------:|:-----------------|
NODE_94_length_44776_cov_27.159388  |    44776 | Phage        |     0.385 |                0.719 | False         | False             |                    2 |                19 |                   0 |                   0 |            0.966 |          1.27  |         3.66  |       1.679 |          -5.832 |         2.477 |          -3.199 |         1.619 | 5V1n14V1n        |
NODE_123_length_36569_cov_24.228077 |    36569 | Phage        |     0.503 |                0.695 | False         | False             |                    1 |                16 |                   0 |                   0 |            0.945 |          0.766 |         3.453 |       1.116 |          -6.02  |         2.471 |          -2.795 |         1.554 | 9V1n7V           |
NODE_149_length_32942_cov_23.754006 |    32942 | Phage        |     0.458 |                0.758 | False         | False             |                    1 |                14 |                   1 |                   0 |           -0.023 |          0.602 |         3.924 |       3.352 |          -7.18  |         5.324 |          -2.023 |         3.229 | 3V2n11V          |
NODE_231_length_24276_cov_21.832294 |    24276 | Phage        |     0.502 |                0.761 | False         | False             |                    2 |                 9 |                   0 |                   0 |            1.08  |          0.978 |         3.297 |       1.479 |          -5.773 |         1.05  |          -2.682 |         1.467 | 1V1n3V1n5V       |
 NODE_262_length_22786_cov_22.465664 |    22786 | Phage        |     0.452 |                0.709 | False         | False             |                    1 |                 9 |                   0 |                   1 |            0.383 |          0.768 |         3.465 |       1.919 |          -6.875 |         1.275 |          -1.683 |         4.078 | 2V1n6V1n1V       |

This table provides information about various contigs in a metagenomic assembly. Each row represents a single contig, and the columns provide information about the contig's ID, length, the number of windows identified as prokaryotic, viral, eukaryotic, and archaeal, the prediction of the contig (Phage or Non-phage), the score of the contig for each category (bacterial, viral, eukaryotic and archaeal), and a summary of the windows. The table can be used to identify potential phage sequences in the metagenomic assembly based on the prediction column. The score columns can be used to further evaluate the confidence of the prediction and the window summary column can be used to understand the count of windows that contributed to the final prediction.

---
### Predicting prophages with Jaeger
---

```
Jaeger -p -i NZ_CP033092.fna -o outdir 
```
![image](https://user-images.githubusercontent.com/34155351/201996217-807c638f-49e1-4147-baff-af6bf20441fa.png)


#### Visualizing predictions 


You can use [phage_contig_annotator](https://github.com/Yasas1994/phage_contig_annotator) to annotate and visualize Jaeger predictions.

---
## Acknowlegements
---

This work was supported by the European Union’s Horizon 2020 research and innovation program, under the Marie Skłodowska-Curie Actions Innovative Training Networks grant agreement no. 955974 ([VIROINF](https://viroinf.eu/)), the European Research Council (ERC) Consolidator grant 865694 

<img src=https://github.com/Yasas1994/Jaeger/assets/34155351/0cad76c6-6e4d-4b89-8e73-257137cf73a8 width="210" height="84"><img src=https://github.com/Yasas1994/Jaeger/assets/34155351/41b3d6e1-709a-4833-a9d7-d69e833797a8 width="100" height="100">

ascii art from  <font size="3"> https://ascii.co.uk/ </font>


