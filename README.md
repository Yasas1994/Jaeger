````
      
      
                       ██╗ █████╗ ███████╗ ██████╗ ███████╗██████╗ 
                       ██║██╔══██╗██╔════╝██╔════╝ ██╔════╝██╔══██╗
                       ██║███████║█████╗  ██║  ███╗█████╗  ██████╔╝
                  ██   ██║██╔══██║██╔══╝  ██║   ██║██╔══╝  ██╔══██╗
                  ╚█████╔╝██║  ██║███████╗╚██████╔╝███████╗██║  ██║
                   ╚════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝



````




# yet Another phaGE identifieR
Identifying phage genome sequences concealed in metagenomes is a long standing problem in viral metagenomics and ecology. Although various methods have been introduced in the recent past, each methods has its own strengths and weaknesses. Recently, machine learning has gained popularity in detecting viruses in metagenome samples. Here, we introduce Jaeger a deep learning based, sensitive phage prediction tool.


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
conda install -n jaeger -c conda-forge cudatoolkit=10.1 

conda install -n jaeger -c conda-forge cudnn=7.6.5.32

conda install -n jaeger tensorflow-gpu=2.4.1 numpy=1.19.5 tqdm=4.64.0 biopython=1.78

````


if you have cuda=11.3 installed on your system,

````
conda install -n jaeger -c anaconda cudatoolkit=11.3 

conda install -n jaeger -c conda-forge cudnn

pip install tensorflow==2.5 numpy==1.19.5 tqdm==4.64.0 biopython==1.78

````


### Setting up the environment (Apple silicon)

````
  conda create -c conda-forge -c apple -c bioconda -c defaults -n jaeger python=3.9.2 tensorflow=2.6 tensorflow-deps=2.6.0 numpy=1.19.5 tqdm=4.64.0 biopython=1.78
````
to add support for apple silicon gpus run

````
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

## What is in the output?


