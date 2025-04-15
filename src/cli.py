#!/usr/bin/env python

"""
Copyright 2024 R. Y. Wijesekara - University Medicine Greifswald, Germany

Identifying phage genome sequences concealed in metagenomes is a
long standing problem in viral metagenomics and ecology.
The Jaeger approach uses homology-free machine learning to identify
both phages and prophages in metagenomic assemblies.
"""

import os
import sys
import click
from importlib.metadata import version
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "darwin":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
elif sys.platform.startswith("linux"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

@click.group()
@click.version_option(version("jaeger-bio"), prog_name="jaeger")
def main():
    "Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences. "


@click.command()
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info", default=1)
def test(**kwargs):
   """Runs tests to check health of the installation. """
   from commands.test import test_core
   test_core(**kwargs)



@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output directory")
@click.option('--fsize', type=int, default=2048, help="Length of the sliding window (value must be 2^n). Default: 2048")
@click.option('--stride', type=int, default=2048, help="Stride of the sliding window. Default: 2048 (stride==fsize)")
@click.option('-m', '--model', type=click.Choice(['default', 'experimental_1', 'experimental_2']), default='default', help="Select a deep-learning model to use. Default: default")
@click.option('-p', '--prophage', is_flag=True, help="Extract and report prophage-like regions. Default: False")
@click.option('-s', '--sensitivity', type=float, default=1.5, help="Sensitivity of the prophage extraction algorithm (0-4). Default: 1.5")
@click.option('--lc', type=int, default=500_000, help="Minimum contig length for prophage extraction. Default: 500000 bp")
@click.option('--rc', type=float, default=0.1, help="Minimum reliability score required to accept predictions. Default: 0.1")
@click.option('--pc', type=int, default=3, help="Minimum phage score required to accept predictions. Default: 3")
@click.option('--batch', type=int, default=96, help="Parallel batch size, lower if GPU runs out of memory. Default: 96")
@click.option('--workers', type=int, default=4, help="Number of threads to use. Default: 4")
@click.option('--getalllogits', is_flag=True, help="Writes window-wise scores to a .npy file")
@click.option('--getsequences', is_flag=True, help="Writes the putative phage sequences to a .fasta file")
@click.option('--cpu', is_flag=True, help="Ignore available GPUs and explicitly run on CPU. Default: False")
@click.option('--physicalid', type=int, default=0, help="Set default GPU device ID for multi-GPU systems. Default: 0")
@click.option('--getalllabels', is_flag=True, help="Get predicted labels for Non-Viral contigs. Default: False")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
@click.option('-f', '--overwrite', is_flag=True, help="Overwrite existing files")
def predict(**kwargs):
    """Run jaeger inference pipeline."""
    from commands.predict import run_core
    run_core(**kwargs)
   
@click.command()
@click.option('-c', '--config', type=click.Path(exists=True, file_okay=True,), required=None, help="Path to tuning configuration file (YAML)")
@click.option('-n', required=True, help="Model name")
@click.option('-o', '--output', type=str, required=True, help="Path to output directory")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def tune(**kwargs):
    """Fine-tune existing models on custom databases."""
    from commands.tune import tune_core
    tune_core(**kwargs)


@click.command()
@click.option('-c', '--config', type=click.Path(exists=True, file_okay=True,), required=True, help="Path to training configuration file (YAML)")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def train(**kwargs):
    """Train new models on custom databases from scratch."""
    from commands.train import train_core
    train_core(**kwargs)

@click.command()
@click.option('-c', '--config', type=click.Path(exists=True, file_okay=True,), required=None, help="Path to training configuration file (YAML)")
@click.option('-m', '--model', type=str, required=True, help="Path to model weights or graph")
@click.option('-n', required=True, help="Model name")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def register_model(**kwargs):
    """Adds newly trained and fine-tuned models to the model database."""
    from commands.train import train_core
    train_core(**kwargs)

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def utils(obj):
    """
    tool chain with auxiliary functions
    """
    pass
@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Shuffles input DNA sequences in .fasta or .txt files to generate
            negative train (validation or train) sets for training (testing)
            out-of-distribution detection models (reliability models)

            usage                                                                      
            -----                                                                      

            jaeger utils shuffle [OPTIONS] -i contigs.fasta -o shuffled_contig.fasta

        """,
)
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output file")
@click.option('--itype', type=click.Choice(['FASTA', 'CSV'], case_sensitive=False), required=True, help="input file type")
@click.option('--otype', type=click.Choice(['FASTA', 'CSV'], case_sensitive=False), required=True, help="out file type")
def dinuc_shuffle(**kwargs):
    """shuffles DNA sequences while preserving the dinucleotide composition."""
    from commands.utils import shuffle_core
    shuffle_core(**kwargs)
    pass



main.add_command(test)
main.add_command(predict)
main.add_command(tune)
main.add_command(train)
main.add_command(utils)


if __name__ == "__main__":
    main()

