````
      
      
                       ██╗ █████╗ ███████╗ ██████╗ ███████╗██████╗ 
                       ██║██╔══██╗██╔════╝██╔════╝ ██╔════╝██╔══██╗
                       ██║███████║█████╗  ██║  ███╗█████╗  ██████╔╝
                  ██   ██║██╔══██║██╔══╝  ██║   ██║██╔══╝  ██╔══██╗
                  ╚█████╔╝██║  ██║███████╗╚██████╔╝███████╗██║  ██║
                   ╚════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝



````



# yet AnothEr phaGE identifieR
Identifying phage genome sequences concealed in metagenomes is a long standing problem in viral metagenomics and ecology. The Jaeger approach uses homology-free machine learning to identify both phages and prophages in metagenomic assemblies.
## Installation 

### Seting up the environment (Linux)
Jaeger is currently tested only on python 3.9.2 therefore, we recomend you to setup a conda environmet by running

##### for CPU only version

````
conda create -n jaeger python=3.9.2

conda install -n jaeger tensorflow=2.4.1 numpy=1.19.5 tqdm=4.64.0 biopython=1.78

````

##### for GPU version

Jaeger workflow can be greatly accelarated on system with gpus.(you can expect linear speed ups by increasing the number of gpus) To add support for gpus, cudatoolkit and cudnn has to installed on the created conda environmnet by running (you can skip this step if you don't wish to use a gpu) 
please specify the cudatoolkit and cudnn versions that are compatible with the installed cuda version on your system.

if you have cuda=10.1 installed on your system, 

````
conda create -n jaeger python=3.9.2

conda install -n jaeger -c conda-forge cudatoolkit=10.1 

conda install -n jaeger -c conda-forge cudnn=7.6.5.32

conda install -n jaeger tensorflow-gpu=2.4.1 numpy=1.19.5 tqdm=4.64.0 biopython=1.78

````


if you have cuda>=11.1 installed on your system,

````
conda create -n jaeger python=3.9.2 

conda install -n jaeger -c "nvidia/label/cuda-11.x.x" cudatoolkit=11.x

conda install -n jaeger -c conda-forge cudnn

conda activate jaeger #or source activate path/to/jaeger/env depending on your cluster configuration

pip install tensorflow==2.5 numpy==1.19.5 tqdm==4.64.0 biopython==1.78

````


### Setting up the environment (Apple silicon)

````
  conda create -c conda-forge -c apple -c bioconda -c defaults -n jaeger python=3.9.2 tensorflow=2.6 tensorflow-deps=2.6.0 numpy=1.19.5 tqdm=4.64.0 biopython=1.78
````
to add support for apple silicon gpus run

````
conda activate jaeger

pip install tensorflow-metal
````


### finally, clone the repository by running

````
git clone https://github.com/Yasas1994/Jaeger
````



## Running Jaeger

Running Jaeger is quite straightforward once the environment is correctly set up. The program accepts both compressed and uncompressed fasta files containing the contigs as input. It outputs a table containing the predictions and various other statics calculated during the runtime. 
By default, Jaeger will run on all the GPUs on the system. This behavior can be overridden by providing a list of GPUs to -gnames option and limiting the number of GPUs availble for jaeger's use.
-ofasta option will write all putative viral contigs (contigs that satisfies the cut-off score) into a separate file

````
python inference.py -i input_file.fasta -o output_file.fasta --batch 128
````
## selecting batch parameter 

You can control the number of parallel computations using this parameter. By default it is set to 512. If you run into OOM errors, please consider setting the --bactch option to a lower value. for example 128 is good enough for a grraphics card with 6 Gb of memory.

### options

````
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to input file
  -o OUTPUT, --output OUTPUT
                        path to output file
  -f [FORMAT], --format [FORMAT]
                        input file format, default: fasta ('genbank','fasta')
  --stride [STRIDE]     stride of the sliding window. default:2048
  --batch [BATCH]       parallel batch size, set to a lower value if your gpu runs out of memory. default:2048
  --cpu                 Number of threads to use for inference. default:False
  --gpu                 run on gpu, runs on all gpus on the system by default: default: True
  --getalllabels        get predicted labels for Non-Viral contigs. default:False
  --meanscore           output mean predictive score per contig. deafault:True
  --fragscore           output percentage of perclass predictions per contig. deafault:True
  -v, --verbose         increase output verbosity
  
````
### Comming soon - Python Library
Jaeger can be integreted into python scripts using the jaegeraa python library as follows.
currenly the predict function accepts 4 diffent input types.
1) Nucleotide sequence -> str
2) List of Nucleotide sequences -> list(str,str,..)
3) python fileobject -> (io.TextIOWrapper)
4) python generator object that yields Nucleotide sequences as str (types.GeneratorType)
5) Biopython Seq object

```
from jaegeraa.api import Predictions

model=Predictor()
predictions=model.predict(input,stride=2048,fragsize=2048,batch=100)

```
```model.predict()``` returns a dictionary of lists in the following format

```
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
```
import pandas as pd
df = DataFrame.from_dict(predictions)
```

## What is in the output?


## Comming soon - Predicting prophages with Jaeger
![image](https://user-images.githubusercontent.com/34155351/201996217-807c638f-49e1-4147-baff-af6bf20441fa.png)


## Visualizing predictions 

You can use [phage_contig_annotator](https://github.com/Yasas1994/phage_contig_annotator) to annotate and visualize Jaeger predictions.



