# Installation

Jaeger can be installed via **one-liner script**, **Bioconda**, **PyPI**, **git**, or **Apptainer/Singularity**. Choose the method that best fits your environment.

---

## Quick reference

| Method | Best for | GPU support | Command |
|--------|----------|-------------|---------|
| [One-liner](#one-liner-recommended) | Quick setup / auto-detection | Yes | `curl \| bash` |
| [Bioconda](#bioconda) | HPC / reproducible environments | Yes | `mamba install jaeger-bio` |
| [PyPI](#pypi) | Development / latest features | Yes | `pip install jaeger-bio[gpu]` |
| [Git](#git) | Contributors / bleeding edge | Yes | `pip install ".[gpu]"` |
| [Apptainer](#apptainer) | Containerized HPC workflows | Yes | `apptainer build …` |

---

## One-liner (recommended)

The easiest way to install Jaeger. Auto-detects your platform and installs the correct variant.

```bash
curl -sSL https://raw.githubusercontent.com/MGXlab/Jaeger/main/install.sh | bash
```

The script will:
1. Detect whether you have an NVIDIA GPU, CPU-only, or Apple Silicon
2. Create a `jaeger` conda environment with Python 3.11–3.13 (or a `jaeger/` venv if conda is unavailable)
3. Install the correct package variant (`[gpu]`, `[cpu]`, or `[darwin-arm]`)
4. Run `jaeger health` to verify the installation

### Next steps

The script installs Jaeger into its own environment, which you must activate before use:

```bash
# conda installation
conda activate jaeger

# or, if the script fell back to a venv
source jaeger/bin/activate
```

Jaeger ships without modern model weights — download the latest model before your first run:

```bash
# List available models (newest first, with upload dates)
jaeger download --list

# Download the latest model (this is a large download, ~1 GB)
jaeger download --path ~/jaeger_models

# Run your first prediction with it
jaeger predict -i contigs.fasta -o results
```

See [Downloading a model](#downloading-a-model) for details.

---

## Bioconda

The simplest way to install Jaeger on most systems. GPU support requires the CUDA Toolkit and cuDNN to be accessible to conda.

```bash
# Add required channels (once)
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create environment and install
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.14" pip jaeger-bio

# Activate
conda activate jaeger

# Verify installation
jaeger health

# Download the latest model
jaeger download --path ~/jaeger_models
```

---

## PyPI

Recommended for users who want the latest stable release or need to install into an existing Python environment.

```bash
# Create a conda environment
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.14" pip
conda activate jaeger

# Or use venv
python3 -m venv jaeger
source jaeger/bin/activate

# Install with GPU support
pip install jaeger-bio[gpu]

# Or CPU-only
pip install jaeger-bio[cpu]

# Or on Apple Silicon (Mac ARM)
pip install jaeger-bio[darwin-arm]

# Verify installation
jaeger health

# Download the latest model
jaeger download --path ~/jaeger_models
```

---

## Git

For developers, contributors, or anyone who needs the very latest code from the `main` branch.

```bash
# Clone the repository
git clone https://github.com/MGXlab/Jaeger.git
cd Jaeger

# Create a conda environment
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.14" pip
conda activate jaeger

# Or use venv
python3 -m venv jaeger
source jaeger/bin/activate

# Install in editable mode
pip install -e ".[gpu]"   # GPU
pip install -e ".[cpu]"   # CPU
pip install -e ".[darwin-arm]"  # Mac ARM

# Download the latest model
jaeger download --path ~/jaeger_models
```

---

## Apptainer

For HPC clusters or environments where you prefer a self-contained container.

```bash
# Download container definition and config
wget -O jaeger_singularity.def https://raw.githubusercontent.com/Yasas1994/Jaeger/main/singularity/jaeger_singularity.def
wget -O config.json https://raw.githubusercontent.com/Yasas1994/Jaeger/main/src/jaeger/data/config.json

# Build the container
apptainer build jaeger.sif singularity/jaeger_singularity.def

# Test the container
apptainer run --nv jaeger.sif jaeger --help
apptainer run --nv jaeger.sif jaeger health

# List available models
apptainer run --nv jaeger.sif jaeger download --list

# Download the latest model
apptainer run --nv jaeger.sif jaeger download \
  --path /path/to/save/model \
  --config /path/to/config.json

# Run prediction
apptainer run --nv jaeger.sif jaeger predict \
  --model jaeger_91c5575c_1.3M_fragment \
  --config /path/to/config.json \
  -i /path/to/input.fasta \
  -o /path/to/save/results
```

---

## Downloading a model

Jaeger bundles only a small legacy model, which is **deprecated**. For real work, download a modern model from the catalog:

```bash
# List available models (newest first, with upload dates)
jaeger download --list

# Download the latest model into a directory of your choice
jaeger download --path ~/jaeger_models

# Or pick a specific model from the list
jaeger download --model_name jaeger_91c5575c_1.3M_fragment --path ~/jaeger_models
```

The model is registered automatically. Select it at prediction time:

```bash
jaeger predict -i contigs.fasta -o results -m jaeger_91c5575c_1.3M_fragment
```

---

## Troubleshooting

### Jaeger fails to detect the GPU

1. **Check CUDA modules** (HPC only):
   ```bash
   module avail
   module load cuda/12.0.0
   ```

2. **Verify NVIDIA drivers**:
   ```bash
   nvidia-smi
   ```

3. **Manually install CUDA toolkit** (if the above fails):
   ```bash
   # Create environment
   conda create -n jaeger -c conda-forge -c bioconda -c defaults "python>=3.11,<3.14" pip

   # Install CUDA and cuDNN
   conda install -n jaeger -c "nvidia/label/cuda-11.8.0" cudatoolkit=11
   conda install -n jaeger -c conda-forge cudnn

   # Install Jaeger
   conda install -n jaeger -c conda-forge -c bioconda jaeger-bio
   conda activate jaeger
   ```

See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for more details on GPU setup.
