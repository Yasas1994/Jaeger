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
import json
import click
from collections import defaultdict
import logging
from pathlib import Path
from importlib.metadata import version
from importlib.resources import files
from jaeger.utils.logging import get_logger
import warnings
warnings.filterwarnings("ignore")

if sys.platform == "darwin":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
elif sys.platform.startswith("linux"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    os.environ["XLA_FLAGS"] ="--xla_gpu_cuda_data_dir=/usr/lib/cuda"

def json_to_dict(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} 

def add_data_to_json(path: str, new_data: dict, list_key: str = None):
    """
    Add new_data into the JSON at `path`.  
    - If the top‐level JSON is a dict, this will merge new_data’s keys.  
    - If list_key is provided, it will append new_data into the list at that key.
    """
    data = json_to_dict(path)

    if list_key:
        # ensure it’s a list
        data.setdefault(list_key, [])
        if not isinstance(data[list_key], list):
            raise ValueError(f"Expected {list_key} to be a list")
        data[list_key].append(new_data)
    else:
        if not isinstance(data, dict):
            raise ValueError("Top‐level JSON is not an object")
        data.update(new_data)

    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

class AvailableModels:
    """
    get all available models from the model path
    """
    def __init__(self, path):
        if isinstance(path, str) or isinstance(path, Path):
            self.paths = [Path(path)]
        elif isinstance(path, list):
            self.paths = [Path(i) for i in path]
        self.info = self._scan_for_models()
        

    def _scan_for_models(self) -> defaultdict:
        _tmp = defaultdict(dict)
        for path in self.paths:
            for model_graph in path.rglob("*_graph"):
                if model_graph.is_dir():
                    _tmp[model_graph.name.rstrip("_graph")]['graph'] = model_graph
                for _match in ("classes", "project"):
                    for _cfg in model_graph.parent.rglob(f"*_{_match}.yaml"):
                        if _cfg.is_file():
                            _tmp[model_graph.name.rstrip("_graph")][_match] = _cfg
        
        return _tmp
USER_MODEL_PATHS = json_to_dict(files('jaeger.data') / "config.json").get("model_paths")
DEFAULT_MODEL_PATH = [files('jaeger.data')]
AVAIL_MODELS = [i for i in AvailableModels(path=DEFAULT_MODEL_PATH + USER_MODEL_PATHS).info.keys() if i != "jaeger_fragment"] + ["default"] 
logger = logging.getLogger("Jaeger")

@click.group()
@click.version_option(version("jaeger-bio"), prog_name="jaeger")
def main():
    "Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences. "


@click.command()
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info", default=1)
def test(**kwargs):
   """Runs tests to check health of the installation. """
   from jaeger.commands.test import test_core
   test_core(**kwargs)



@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output directory")
@click.option('--fsize', type=int, default=2048, help="Length of the sliding window (value must be 2^n). Default: 2048")
@click.option('--stride', type=int, default=2048, help="Stride of the sliding window. Default: 2048 (stride==fsize)")
@click.option('-m', '--model', type=click.Choice(AVAIL_MODELS), default='default', help="Select a deep-learning model to use. Default: default")
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
    if kwargs.get('model') == "default":
        from jaeger.commands.predict_legacy import run_core
        run_core(**kwargs)
    else:
        logger.info("Comming soon!")
   
@click.command()
@click.option('-c', '--config', type=click.Path(exists=True, file_okay=True,), required=None, help="Path to tuning configuration file (YAML)")
@click.option('-n', required=True, help="Model name")
@click.option('-o', '--output', type=str, required=True, help="Path to output directory")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def tune(**kwargs):
    """Fine-tune existing models on custom databases."""
    from jaeger.commands.tune import tune_core
    tune_core(**kwargs)


@click.command()
@click.option('-c', '--config', type=click.Path(exists=True, file_okay=True,), required=True, help="Path to training configuration file (YAML)")
@click.option('--only_classification_head', is_flag=True, required=False, help="Only update the classification head")
@click.option('--from_last_checkpoint', is_flag=True, required=False, help="Start training from the last checkpoint")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def train(**kwargs):
    """Train new models on custom databases from scratch."""
    from jaeger.commands.train import train_contig_core, train_fragment_core
    train_fragment_core(**kwargs)

@click.command()
@click.option('-p', '--path', type=str, required=True, help="Path to model weights, graph and configuration files")
@click.option('-v', '--verbose', count=True, help="Verbosity level: -vv debug, -v info (default: info)", default=1)
def register_models(**kwargs):
    """Appends newly trained and fine-tuned models to the model path"""
 

    path = Path(files('jaeger.data')) / "config.json"
    add_data_to_json(path, kwargs.get("path"), list_key="model_paths")

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
    from jaeger.commands.utils import shuffle_core
    shuffle_core(**kwargs)


@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Simulate a metagenome assembly from contigs/genomes given fragment size range.

            usage                                                                      
            -----                                                                      

            jaeger utils fragment [OPTIONS] -i contigs.fasta -o fragemented_contigs.fasta

        """,
)
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output file")
@click.option('--minlen', type=int, required=True, help="min fragment size")
@click.option('--maxlen', type=int, required=True, help="max fragment size")
@click.option('--overlap', type=int, required=False, help="overlap between fragments", default=0)
@click.option('--shuffle', is_flag=True, help="enable shuffling.")
def fragment(**kwargs):
    """shuffles DNA sequences while preserving the dinucleotide composition."""
    from jaeger.commands.utils import split_core
    split_core(**kwargs)


@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Gradually mask random positions in a given fasta record.

            usage                                                                      
            -----                                                                      

            jaeger utils mask [OPTIONS] -i contigs.fasta -o fragemented_contigs.fasta

        """,
)
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output file")
@click.option('--minperc', type=float, required=False, help="min percentage of positions to mask", default=0.0)
@click.option('--maxperc', type=float, required=False, help="maxp percentage of positions to mask", default=1.0)
@click.option('--step', type=float, required=False, help="perc positions to mask in each iteration", default=0.01)
def mask(**kwargs):
    """shuffles DNA sequences while preserving the dinucleotide composition."""
    from jaeger.commands.utils import mask_core
    mask_core(**kwargs)



@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Generate a non-redundant fragment database from fasta file for training/validating
            fragment models

            usage                                                                      
            -----                                                                      

            jaeger utils dataset [OPTIONS] -i contigs.fasta -o fragemented_contigs.fasta

        """,
)
@click.option('-i', '--input', type=click.Path(exists=True), required=True, help="Path to input file")
@click.option('-o', '--output', type=str, required=True, help="Path to output file")
@click.option('--class',  type=int, required=True, help="class label")
@click.option('--valperc', type=float, required=False, help="percentage of fragments to include in validation set", default=0.1)
@click.option('--trainperc', type=float, required=False, help="percentage of fragments to include in train set", default=0.8)
@click.option('--testperc', type=float, required=False, help="percentage of fragments to include in test set", default=0.1)
@click.option('--maxiden',  type=float, required=False, help="max identity between any two fragments", default=0.6)
@click.option('--maxcov',  type=float, required=False, help="max coverage between any two fragments", default=0.6)
@click.option('--fraglen',  type=int, required=False, help="max fragment length", default=2048)
@click.option('--overlap',  type=int, required=False, help="max overlap between sequences", default=1024)
@click.option('--outtype', type=click.Choice(['CSV', 'FASTA'], case_sensitive=False), required=False, help="output type", default="CSV")
@click.option('--method', type=click.Choice(['ANI', 'AAI'], case_sensitive=False), required=False, help="dereplication method", default="ANI")

def dataset(**kwargs):
    """Generate a non-redundant fragment database from fasta file for training/validating
            fragment models"""
    from jaeger.commands.utils import dataset_core
    dataset_core(**kwargs)
    pass


main.add_command(test)
main.add_command(predict)
main.add_command(tune)
main.add_command(train)
main.add_command(register_models)
main.add_command(utils)


if __name__ == "__main__":
    main()

