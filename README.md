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



Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences
===============
![GitHub](https://img.shields.io/github/license/Yasas1994/jaeger) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Yasas1994/jaeger/main?color=8a35da) ![Conda](https://img.shields.io/conda/v/bioconda/jaeger-bio) ![Conda](https://img.shields.io/conda/dn/bioconda/jaeger-bio) [![PyPI version](https://img.shields.io/pypi/v/jaeger-bio.svg)](https://pypi.org/project/jaeger-bio/) [![Downloads](https://static.pepy.tech/badge/jaeger-bio)](https://pepy.tech/project/jaeger-bio)



Jaeger is a tool that utilizes homology-free machine learning to identify phage genome sequences that are hidden within metagenomes. It is capable of detecting both phages and prophages within metagenomic assemblies.

---
#### Citing Jaeger
---

If you use Jaeger in your work, please consider citing its preprint: 

* **[Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences](https://www.biorxiv.org/content/early/2024/09/24/2024.09.24.612722)**  
  Yasas Wijesekara, Ling-Yi Wu, Rick Beeloo, Piotr Rozwalak, Ernestina Hauptfeld, Swapnil P. Doijad, Bas E. Dutilh, Lars Kaderali  
  *bioRxiv*, 2024.09.24.612722



To cite the code itself:

* **Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences**  
https://doi.org/10.5281/zenodo.20534106

---

- [Installing Jaeger](#installation)
  - [Bioconda](#option-1--bioconda)
  - [PyPi](#option-2--installing-from-pypi)
  - [git (main branch)](#option-3--installing-from-git)
  - [containers](#option-4-building-a-singularity-container)
- [Troubleshooting](#troubleshooting)
- [Running Jaeger](#running-jaeger)
- [What is in the output](#what-is-in-the-output)
- [Predicting prophages](#predicting-prophages-with-jaeger)

--- 
#### Installing Jaeger
---

##### option 1 : bioconda

The performance of the Jaeger workflow can be significantly increased by utilizing GPUs. To enable GPU support, the CUDA Toolkit and cuDNN library must be accessible to conda.

````bash
# create conda environment and install jaeger
mamba create -n jaeger -c bioconda jaeger-bio==1.2

# activate environment
conda activate jaeger
````
Test the installation with test data
```bash
jaeger health
```


##### option 2 : Installing from PyPI (recommended)

```bash
# create a conda environment and activate  
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<=3.12" pip
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

# test the installation
jaeger health
```

##### option 3 : Installing from git (main branch)

```bash
# create a conda environment and activate  
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.12" pip
conda activate jaeger

# OR create a virtual environment using venv
python3 -m venv jaeger
source jaeger/bin/activate    

# install jaeger

# to install with GPU support
pip install --no-cache-dir "jaeger-bio[gpu] @ git+https://github.com/MGXlab/Jaeger@main"

# to install without GPU support
pip3 install --root-user-action=ignore --no-cache-dir "jaeger-bio[cpu] @ git+https://github.com/MGXlab/Jaeger@main"

# to install on a Mac(arm)
pip3 install --root-user-action=ignore --no-cache-dir "jaeger-bio[darwin-arm] @ git+https://github.com/MGXlab/Jaeger@main"

# test the installation
jaeger health

```

##### option 4 : Apptainer (Singularity)
If you're using Apptainer on a cluster, it's recommended to build the container on your local machine and then transfer it to the cluster.
```bash
# get the container def
wget -O jaeger_singularity.def https://raw.githubusercontent.com/Yasas1994/Jaeger/main/singularity/jaeger_singularity.def
# get the configuration file
wget -O config.json https://raw.githubusercontent.com/Yasas1994/Jaeger/main/src/jaeger/data/config.json

# to build the container
apptainer build jaeger.sif singularity/jaeger_singularity.def

# test container
apptainer run --nv jaeger.sif jaeger --help

# test the installation
apptainer run --nv jaeger.sif jaeger health

# list jaeger models available for download
apptainer run --nv jaeger.sif jaeger download --list
# download jaeger models
apptainer run --nv jaeger.sif jaeger download --model_name jaeger_57341_1.5M_fragment --path /path/to/save/model --config /path/to/config.json

# run jaeger
apptainer run --nv jaeger.sif jaeger predict --model jaeger_57341_1.5M_fragment --config /path/to/config.json -i /path/to/input.fasta -o /path/to/save/results

```


---
#### Downloading models
---
Starting from version 1.2.0, users will need to download the new models separately after installing Jaeger. However, for backward compatibility, Jaeger will still include the old model by default.

Use the --list flag to print out all models available for download
```bash
jaeger download --list
```
Then to download the model and add it to the model path run
```bash
jaeger download --path /path/to/store/models --model_name jaeger_38341_1.4M
```
If you decide to change the model path later, or if you have a directory with newly trained/tuned models
register the path 
```bash
jaeger register-models --path /new/model/path
```


---
#### Running Jaeger
---
##### CPU/GPU mode
Once the environment is properly set up, using Jaeger is straightforward. The program can accept both compressed and uncompressed .fasta files containing the contigs as input. It will output a table containing the predictions and various statistics calculated during runtime. 

```bash
jaeger predict -i input_file.fasta -o output_dir --batch 128
```

To run jaeger with Apptainer/Singularity
```bash
apptainer run --nv jaeger.sif jaeger predict -i input_file.fasta -o output_dir --batch 128
```


##### Selecting the batch parameter 

You can control the number of parallel computations using this parameter. By default it is set to 96. If you run into OOM errors, please consider setting the --batch option to a lower value. For example, 96 is good enough for a graphics card with 4 GB of memory.

---
#### What is in the output?
---
All predictions are summarized in a table located at ```output_dir/<input_file>_jaeger.tsv```

```
┌───────────────────────────────────┬────────┬────────────┬─────────┬───────────────────┬─────────────┬─────────────┬───┬─────────────┬────────────────┬──────────────────┬───────────────┐
│ contig_id                         ┆ length ┆ prediction ┆ entropy ┆ reliability_score ┆ host_contam ┆ prophage_contam ┆ … ┆ window_summary ┆ terminal_repeats ┆ repeat_length │
╞═══════════════════════════════════╪════════╪════════════╪═════════╪═══════════════════╪═════════════╪═════════════╪═══╪═════════════╪════════════════╪══════════════════╪═══════════════╡
│ NODE_1109_length_9622_cov_23.163… ┆ 9622   ┆ phage      ┆ 0.127   ┆ 0.730             ┆ False       ┆ False       ┆ … ┆ 1V1n2V         ┆ DTR              ┆ 13            │
│ NODE_1181_length_9275_cov_26.864… ┆ 9275   ┆ phage      ┆ 0.203   ┆ 0.723             ┆ False       ┆ False       ┆ … ┆ 4V             ┆ null             ┆ null          │
│ NODE_123_length_36569_cov_24.228… ┆ 36569  ┆ phage      ┆ 0.458   ┆ 0.702             ┆ False       ┆ False       ┆ … ┆ 9V1n7V         ┆ null             ┆ null          │
│ NODE_149_length_32942_cov_23.754… ┆ 32942  ┆ phage      ┆ 0.503   ┆ 0.691             ┆ False       ┆ False       ┆ … ┆ 3V1n1n11V      ┆ null             ┆ null          │
│ NODE_231_length_24276_cov_21.832… ┆ 24276  ┆ phage      ┆ 0.502   ┆ 0.688             ┆ False       ┆ False       ┆ … ┆ 1V1n3V1n5V     ┆ null             ┆ null          │
└───────────────────────────────────┴────────┴────────────┴─────────┴───────────────────┴─────────────┴─────────────┴───┴─────────────┴────────────────┴──────────────────┴───────────────┘

```

This table provides information about various contigs in a metagenomic assembly. Each row represents a single contig, and the columns provide information about the contig's ID, length, prediction (phage or bacteria), entropy of the softmax distribution, reliability score, host/prophage contamination flags, per-class window counts and scores, and a summary of the windows. The `reliability_score` (0–1, higher is more confident) and `entropy` (0–2, lower is more confident) can be used to evaluate prediction confidence. The `window_summary` column shows the pattern of phage (V) and non-phage (n) windows along the contig. The `terminal_repeats` and `repeat_length` columns report detected terminal repeat structures (e.g., DTR = direct terminal repeats).

---

#### Options
---

````
jaeger predict --help
````
````

Usage: jaeger predict [OPTIONS]

  Runs Jaeger on a dataset

Options:
  -i, --input PATH         Path to input file  [required]
  -o, --output TEXT        Path to output directory  [required]
  --fsize INTEGER          Length of the sliding window  [default: 2000]
  --stride INTEGER         The gap between two the sliding windows
                           (stride==fsize)  [default: 2000]
  -m, --model TEXT         Select a deep-learning model to use.  [default:
                           default]
  --model_path TEXT        Give the path to a model. overrides --model
  --config PATH            Path to Jaeger config file (e.g., when using
                           Apptainer or Docker)
  -p, --prophage           Extract and report prophage-like regions
  -s, --sensitivity FLOAT  Sensitivity of the prophage extraction algorithm
                           (0-4)  [default: 1.5]
  --lc INTEGER             Minimum contig length for prophage extraction
                           [default: 500000]
  --rc FLOAT               Minimum reliability score required to accept
                           predictions  [default: 0.1]
  --pc INTEGER             Minimum phage score required to accept predictions
                           [default: 3]
  --batch INTEGER          Parallel batch size, lower if GPU runs out of
                           memory  [default: 96]
  --workers INTEGER        Number of threads to use  [default: 4]
  --getalllogits           Writes window-wise scores to a .npy file
  --getsequences           Writes the putative phage sequences to a .fasta
                           file
  --cpu                    Ignore available GPUs and explicitly run on CPU
  --physicalid INTEGER     Set default GPU device ID for multi-GPU systems
                           [default: 0]
  --mem INTEGER            GPU memory limit  [default: 4]
  --getalllabels           Get predicted labels for Non-Viral contigs
  -v, --verbose            Verbosity level: -vv debug, -v info  [default: 1]
  -f, --overwrite          Overwrite existing files
  --help                   Show this message and exit.


````
---
#### Python Library
---

> **Note:** The Python API (`jaeger.api`) is currently experimental and not available in the latest release. For programmatic access, we recommend using the CLI or calling Jaeger via `subprocess`.
>
> ```python
> import subprocess
> subprocess.run([
>     "jaeger", "predict",
>     "-i", "input.fasta",
>     "-o", "output_dir",
>     "--batch", "128"
> ])
> ```
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
jaeger predict -p -i NC_002695.fna -o outdir 
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

Users can find the following visualization in the `plots` directory <br><br>

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
#### Acknowledgements
---

This work was supported by the European Union’s Horizon 2020 research and innovation program, under the Marie Skłodowska-Curie Actions Innovative Training Networks grant agreement no. 955974 ([VIROINF](https://viroinf.eu/)), the European Research Council (ERC) Consolidator grant 865694 

<img src=https://github.com/Yasas1994/Jaeger/assets/34155351/0cad76c6-6e4d-4b89-8e73-257137cf73a8 width="210" height="84">  &nbsp;&nbsp;&nbsp;  <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/fef3bc35-8a8c-44c9-85ca-35ab0c68130e width="100" height="100">  &nbsp;&nbsp;&nbsp;   <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/f15ab9b6-cade-4315-941c-e897f753dad9 width="150" height="100">

The ascii art logo is from  <font size="3"> https://ascii.co.uk/ </font>


