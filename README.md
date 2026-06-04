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
        ,'`.  \/ |=| \/  ,'`.```

```



Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences
===============
![GitHub](https://img.shields.io/github/license/Yasas1994/jaeger) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Yasas1994/jaeger/main?color=8a35da) ![Conda](https://img.shields.io/conda/v/bioconda/jaeger-bio) ![Conda](https://img.shields.io/conda/dn/bioconda/jaeger-bio) [![PyPI version](https://img.shields.io/pypi/v/jaeger-bio.svg)](https://pypi.org/project/jaeger-bio/) [![Downloads](https://static.pepy.tech/badge/jaeger-bio)](https://pepy.tech/project/jaeger-bio)



Jaeger is a tool that utilizes homology-free machine learning to identify phage genome sequences that are hidden within metagenomes. It is capable of detecting both phages and prophages within metagenomic assemblies.

> 📚 **For detailed installation instructions, usage guides, and troubleshooting, please visit the [documentation](https://jaeger.readthedocs.io/en/latest/installation.html).**

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
- [Downloading models](#downloading-models)

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
#### Visualizing predictions
---

You can use [phage_contig_annotator](https://github.com/Yasas1994/phage_contig_annotator) to annotate and visualize Jaeger predictions.

---
#### Acknowledgements
---

This work was supported by the European Union's Horizon 2020 research and innovation program, under the Marie Skłodowska-Curie Actions Innovative Training Networks grant agreement no. 955974 ([VIROINF](https://viroinf.eu/)), the European Research Council (ERC) Consolidator grant 865694

<img src=https://github.com/Yasas1994/Jaeger/assets/34155351/0cad76c6-6e4d-4b89-8e73-257137cf73a8 width="210" height="84">  &nbsp;&nbsp;&nbsp;  <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/fef3bc35-8a8c-44c9-85ca-35ab0c68130e width="100" height="100">  &nbsp;&nbsp;&nbsp;   <img src=https://github.com/Yasas1994/Jaeger/assets/34155351/f15ab9b6-cade-4315-941c-e897f753dad9 width="150" height="100">

The ascii art logo is from  <font size="3"> https://ascii.co.uk/ </font>
