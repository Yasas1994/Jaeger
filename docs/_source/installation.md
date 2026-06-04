# Installation

Jaeger can be installed via **Bioconda**, **PyPI**, **git**, or **Apptainer/Singularity**. Choose the method that best fits your environment.

::::{tab-set}

:::{tab-item} One-liner (recommended)

The easiest way to install Jaeger is using the one-liner install script. It auto-detects your platform (GPU, CPU, or Apple Silicon) and installs the correct variant.

```bash
curl -sSL https://raw.githubusercontent.com/MGXlab/Jaeger/main/install.sh | bash
```

The script will:
1. Detect whether you have an NVIDIA GPU, CPU-only, or Apple Silicon
2. Create a `jaeger` conda environment with Python 3.11–3.12
3. Install the correct package variant (`[gpu]`, `[cpu]`, or `[darwin-arm]`)
4. Run `jaeger health` to verify the installation

:::

:::{tab-item}  bioconda

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
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.13" pip jaeger-bio

# Activate
conda activate jaeger

# Verify installation
jaeger health
```

---

## PyPI

Recommended for users who want the latest stable release or need to install into an existing Python environment.

```bash
# Create a conda environment
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.13" pip
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
```

---

## Git

For developers, contributors, or anyone who needs the very latest code from the `main` branch.

```bash
# Clone the repository
git clone https://github.com/MGXlab/Jaeger.git
cd Jaeger

# Create a conda environment
mamba create -n jaeger -c nvidia -c conda-forge cuda-nvcc "python>=3.11,<3.13" pip
conda activate jaeger

# Or use venv
python3 -m venv jaeger
source jaeger/bin/activate

# Install in editable mode
pip install -e ".[gpu]"   # GPU
pip install -e ".[cpu]"   # CPU
pip install -e ".[darwin-arm]"  # Mac ARM
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

# Download a model
apptainer run --nv jaeger.sif jaeger download \
  --model_name jaeger_57341_1.5M_fragment \
  --path /path/to/save/model \
  --config /path/to/config.json

# Run prediction
apptainer run --nv jaeger.sif jaeger predict \
  --model jaeger_57341_1.5M_fragment \
  --config /path/to/config.json \
  -i /path/to/input.fasta \
  -o /path/to/save/results
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
   conda create -n jaeger -c conda-forge -c bioconda -c defaults "python>=3.11,<3.13" pip

   # Install CUDA and cuDNN
   conda install -n jaeger -c "nvidia/label/cuda-11.8.0" cudatoolkit=11
   conda install -n jaeger -c conda-forge cudnn

   # Install Jaeger
   conda install -n jaeger -c conda-forge -c bioconda jaeger-bio
   conda activate jaeger
   ```

See the [TensorFlow installation guide](https://www.tensorflow.org/install/pip) for more details on GPU setup.
