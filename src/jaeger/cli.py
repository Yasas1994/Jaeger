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
import logging
from pathlib import Path
from importlib.metadata import version
from importlib.resources import files
from jaeger.utils.misc import json_to_dict, add_data_to_json, AvailableModels
import warnings

warnings.filterwarnings("ignore")

if sys.platform == "darwin":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
elif sys.platform.startswith("linux"):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"


USER_MODEL_PATHS = json_to_dict(files("jaeger.data") / "config.json").get("model_paths")
DEFAULT_MODEL_PATH = [files("jaeger.data")]
AVAIL_MODELS = [
    i
    for i in AvailableModels(path=DEFAULT_MODEL_PATH + USER_MODEL_PATHS).info.keys()
    if i != "jaeger_fragment"
] + ["default"]
logger = logging.getLogger("Jaeger")


@click.group()
@click.version_option(version("jaeger-bio"), prog_name="jaeger")
def main():
    "Jaeger: an accurate and fast deep-learning tool to detect bacteriophage sequences."


@click.command()
@click.option(
    "-v", "--verbose", count=True, help="Verbosity level: -vv debug, -v info", default=1
)
def test(**kwargs):
    """Runs tests to check health of the installation."""
    from jaeger.commands.test import test_core

    test_core(**kwargs)


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2048,
    help="Length of the sliding window (value must be 2^n). Default: 2048",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="Stride of the sliding window. Default: 2048 (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Default: 'default'. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "-p",
    "--prophage",
    is_flag=True,
    help="Extract and report prophage-like regions. Default: False",
)
@click.option(
    "-s",
    "--sensitivity",
    type=float,
    default=1.5,
    help="Sensitivity of the prophage extraction algorithm (0-4). Default: 1.5",
)
@click.option(
    "--lc",
    type=int,
    default=500_000,
    help="Minimum contig length for prophage extraction. Default: 500000 bp",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions. Default: 0.1",
)
@click.option(
    "--pc",
    type=int,
    default=3,
    help="Minimum phage score required to accept predictions. Default: 3",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory. Default: 96",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use. Default: 4"
)
@click.option(
    "--getalllogits", is_flag=True, help="Writes window-wise scores to a .npy file"
)
@click.option(
    "--getsequences",
    is_flag=True,
    help="Writes the putative phage sequences to a .fasta file",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU. Default: False",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems. Default: 0",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit: Default: 4G",
)
@click.option(
    "--getalllabels",
    is_flag=True,
    help="Get predicted labels for Non-Viral contigs. Default: False",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files")
def predict(**kwargs):
    model = kwargs.get("model")
    if kwargs.get("config") is None:
        if model is None:
            model = "default"
        elif model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )
    else:
        if model is None:
            model = "default"

    """Run jaeger inference pipeline."""
    if model == "default":
        from jaeger.commands.predict_legacy import run_core

        run_core(**kwargs)
    else:
        from jaeger.commands.predict import run_core

        run_core(**kwargs)


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(
        exists=True,
        file_okay=True,
    ),
    required=None,
    help="Path to tuning configuration file (YAML)",
)
@click.option("-n", required=True, help="Model name")
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
def tune(**kwargs):
    """Fine-tune existing models on custom databases."""
    from jaeger.commands.tune import tune_core

    tune_core(**kwargs)


@click.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(
        exists=True,
        file_okay=True,
    ),
    required=True,
    help="Path to training configuration file (YAML)",
)
@click.option(
    "--only_classification_head",
    is_flag=True,
    required=False,
    help="Only train the classification head without the updating the representation learner's weights",
)
@click.option(
    "--only_reliability_head",
    is_flag=True,
    required=False,
    help="Only train the reliability model head",
)
@click.option(
    "--self_supervised_pretraining",
    is_flag=True,
    required=False,
    help="retrain representation model with self supervised learning",
)
@click.option(
    "--only_heads",
    is_flag=True,
    required=False,
    help="Only train the reliability model head",
)
@click.option(
    "--from_last_checkpoint",
    is_flag=True,
    required=False,
    help="continue training from the last checkpoints",
)
@click.option(
    "--only_save",
    is_flag=True,
    required=False,
    help="save the model with weights from last checkpoints without training",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
def train(**kwargs):
    """Train new models on custom databases from scratch.

    # to start training from scratch
    jaeger train -c training/test8.yaml

    # to resume training (representation learner and heads) from latest checkpoints
    jaeger train -c training/test8.yaml --from_last_checkpoint

    # to resume training the classification head from the latest classification model checkpoint
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_classification_head

    # to resume training the reliability model head from the latest reliability model checkpoint
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_reliability_head


    # to resume training the reliability model head and classification model head from the latest model checkpoints
    jaeger train -c training/test8.yaml --from_last_checkpoint --only_heads

    """

    # Collect which flags are True
    selected = [
        flag
        for flag, value in {
            "only_classification_head": kwargs.get("only_classification_head", False),
            "only_reliability_head": kwargs.get("only_reliability_head", False),
            "only_heads": kwargs.get("only_heads", False),
        }.items()
        if value
    ]

    # Check mutual exclusivity
    if len(selected) > 1:
        raise click.UsageError(
            f"Options {', '.join('--' + flag for flag in selected)} are mutually exclusive. Please specify only one."
        )
    from jaeger.commands.train import train_fragment_core

    train_fragment_core(**kwargs)


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(file_okay=False),
    required=True,
    help="Path to model weights, graph and configuration files",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to jager config file (Apptainer or Docker)",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
def register_models(**kwargs):
    """Appends newly trained and fine-tuned models to the model path"""

    if kwargs.get("config") is not None:
        # update an external config file
        path = Path(kwargs.get("config"))
    else:
        # update internal config file
        path = Path(files("jaeger.data")) / "config.json"
    model_path = Path(kwargs.get("path")).resolve()
    add_data_to_json(path, str(model_path), list_key="model_paths")


@click.command()
@click.option(
    "-p",
    "--path",
    type=click.Path(file_okay=False),
    required=False,
    help="Path to save model weights, graph and configuration files",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to jager config file (Apptainer or Docker)",
)
@click.option(
    "-m", "--model", "model_name", required=False, help="Name of the model to download"
)
@click.option(
    "-l",
    "--list",
    "list_models",
    is_flag=True,
    default=False,
    help="List all available models",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
def download(path, model_name, list_models, **kwargs):
    """Downloads model weights and updated model paths, or lists available models."""
    from jaeger.commands.downloads import list_ckan_model_download_links, download_file

    # Enforce mutual exclusivity
    if list_models and (model_name or path):
        raise click.UsageError(
            "The '--list' option cannot be used with '--model' or '--path'."
        )

    if not list_models and (not model_name or not path):
        raise click.UsageError(
            "You must provide both '--model' and '--path', or just '--list'."
        )

    model_links = list_ckan_model_download_links()

    if list_models:
        click.echo("Available models:")
        for name in sorted(model_links.keys()):
            click.echo(f"- {name}")
        return

    if model_name not in model_links:
        raise click.UsageError(
            f"Model '{model_name}' not found. Use '--list' to see available models."
        )

    # to avoid jaeger from scanning huge dirs for models
    model_path = Path(path).resolve() / "jaeger_models"
    model_path.mkdir(parents=True, exist_ok=True)

    download_file((model_name, model_links[model_name]), output_dir=model_path)

    if kwargs.get("config", False):
        # update an external config file
        config_path = Path(kwargs.get("config"))
    else:
        # update internal config file
        config_path = Path(files("jaeger.data")) / "config.json"
    try:
        add_data_to_json(config_path, str(model_path), list_key="model_paths")
    except Exception:
        logger.warning(
            "failed to add model path to jaeger config. Seems like you are running jaeger inside a container. please explicitly define the model path in a config file `jaeger predict --config " \
            "wget -O config.json https://raw.githubusercontent.com/Yasas1994/Jaeger/dev/src/jaeger/data/config.json"
        )


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
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output file")
@click.option("--dinuc", is_flag=True, required=False, help="dinuc shuffle")
@click.option("-k", type=int, default=1, required=False, help="kmer size" )
@click.option(
    "--itype",
    type=click.Choice(["FASTA", "CSV"], case_sensitive=False),
    required=True,
    help="input file type",
)
@click.option(
    "--otype",
    type=click.Choice(["FASTA", "CSV"], case_sensitive=False),
    required=True,
    help="out file type",
)
def shuffle(**kwargs):
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
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output file")
@click.option("--minlen", type=int, required=True, help="min fragment size")
@click.option("--maxlen", type=int, required=True, help="max fragment size")
@click.option(
    "--overlap", type=int, required=False, help="overlap between fragments", default=0
)
@click.option("--shuffle", is_flag=True, help="enable shuffling.")
def fragment(**kwargs):
    """shuffles DNA sequences while preserving the dinucleotide composition."""
    from jaeger.commands.utils import split_core

    split_core(**kwargs)


@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Gradually mask or mutate random positions in a given fasta file.

            usage                                                                      
            -----                                                                      

            jaeger utils mask [OPTIONS] -i contigs.fasta -o fragemented_contigs.fasta
            
            "CACTACACGTACG" -> "cACTacACGtacg"

            jaeger utils mask [OPTIONS] -i contigs.fasta -o fragemented_contigs.fasta --mutate
            
            "CACTACACGTACG" -> "TACTGAACGTTGT"
        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output file")
@click.option(
    "--minperc",
    type=float,
    required=False,
    help="min percentage of positions to mask",
    default=0.0,
)
@click.option(
    "--maxperc",
    type=float,
    required=False,
    help="maxp percentage of positions to mask",
    default=1.0,
)
@click.option(
    "--step",
    type=float,
    required=False,
    help="perc positions to mask in each iteration",
    default=0.01,
)
@click.option(
    "--mutate",
    is_flag=True,
    required=False,
    help="replace with random nucleotides without masking",
    default=False,
)
def mask(**kwargs):
    """shuffles DNA sequences while preserving the dinucleotide composition."""
    from jaeger.commands.utils import mask_core

    if kwargs.get('maxperc') > kwargs.get('minperc'):
        raise click.BadParameter("maxperc can not be lower than minperc")

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
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output file")
@click.option("--class", type=int, required=True, help="class label")
@click.option(
    "--valperc",
    type=float,
    required=False,
    help="percentage of fragments to include in validation set",
    default=0.1,
)
@click.option(
    "--trainperc",
    type=float,
    required=False,
    help="percentage of fragments to include in train set",
    default=0.8,
)
@click.option(
    "--testperc",
    type=float,
    required=False,
    help="percentage of fragments to include in test set",
    default=0.1,
)
@click.option(
    "--maxiden",
    type=float,
    required=False,
    help="max identity between any two fragments",
    default=0.6,
)
@click.option(
    "--maxcov",
    type=float,
    required=False,
    help="max coverage between any two fragments",
    default=0.6,
)
@click.option(
    "--fraglen", type=int, required=False, help="max fragment length", default=2048
)
@click.option(
    "--overlap",
    type=int,
    required=False,
    help="max overlap between sequences",
    default=1024,
)
@click.option(
    "--outtype",
    type=click.Choice(["CSV", "FASTA"], case_sensitive=False),
    required=False,
    help="output type",
    default="CSV",
)
@click.option(
    "--method",
    type=click.Choice(["ANI", "AAI"], case_sensitive=False),
    required=False,
    help="dereplication method",
    default="ANI",
)
@click.option(
    "--ood",
    is_flag=True,
    required=False,
    help="generate dataset for training ood model",
    default=False,
)
def dataset(**kwargs):
    """Generate a non-redundant fragment database from fasta file for training/validating
    fragment models"""
    from jaeger.commands.utils import dataset_core

    dataset_core(**kwargs)
    pass

@utils.command(
    context_settings=dict(ignore_unknown_options=True),
    help="""
            Convert training data between CSV and FASTA formats

            usage                                                                      
            -----                                                                      

            jaeger utils convert -itype FASTA -i contigs.fasta -o fragemented_contigs.csv
            
        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output file")
@click.option(
    "--itype",
    type=click.Choice(["FASTA", "CSV"], case_sensitive=False),
    required=True,
    help="input file type",
)
def convert(**kwargs):
    """convert CSV to FASTA and FASTA to CSV"""
    from jaeger.commands.utils import convert_core

    convert_core(**kwargs)
    pass

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def taxonomy(obj):
    """
    exprimental taxonomy prediction pipeline
    """
    pass
@taxonomy.command(context_settings=dict(ignore_unknown_options=True),
    help="""
            Build a taxonomy database

            usage                                                                      
            -----                                                                      

            jaeger taxonomy build [OPTIONS] -i contigs.fasta -o taxonomy_db

        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-t",
    "--tax",
    type=click.Path(exists=True),
    required=True,
    help="Path to taxdump",
)
@click.option(
    "-a",
    "--acc2tax",
    type=click.Path(exists=True),
    required=True,
    help="Path to accession2taxid file",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2048,
    help="Length of the sliding window (value must be 2^n). Default: 2048",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="Stride of the sliding window. Default: 2048 (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Default: 'default'. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions. Default: 0.1",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory. Default: 96",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use. Default: 4"
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU. Default: False",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems. Default: 0",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit: Default: 4G",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing database")
def build(**kwargs):
    model = kwargs.get("model")
    if kwargs.get("config") is None:
        if model is None:
            model = "default"
        elif model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )
    else:
        if model is None:
            model = "default"

    """Run jaeger taxonomy database generation pipeline"""
    if model == "default":
        raise click.BadParameter(
            f"Model '{model}' is not supported"
        )

    else:
        from jaeger.commands.taxonomy import build_taxdb
        build_taxdb(**kwargs)

@taxonomy.command('predict', context_settings=dict(ignore_unknown_options=True),
    help="""
            Use exeperimental taxonomy prediction workflow

            usage                                                                      
            -----                                                                      

            jaeger taxonomy predict [OPTIONS] -i contigs.fasta -d taxonomy_db -o output_dir

        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file",
)
@click.option(
    "-d",
    "--db",
    type=click.Path(exists=True),
    required=True,
    help="Path to taxonomy database",
)
@click.option(
    "-o", "--output", type=str, required=True, help="Path to output directory"
)
@click.option(
    "--fsize",
    type=int,
    default=2048,
    help="Length of the sliding window (value must be 2^n). Default: 2048",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="Stride of the sliding window. Default: 2048 (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Default: 'default'. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (e.g., when using Apptainer or Docker)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions. Default: 0.1",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory. Default: 96",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use. Default: 4"
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU. Default: False",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems. Default: 0",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit: Default: 4G",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info (default: info)",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing database")
def predict_tax(**kwargs):  # noqa: F811
    model = kwargs.get("model")
    if kwargs.get("config") is None:
        if model is None:
            model = "default"
        elif model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )
    else:
        if model is None:
            model = "default"

    """Run jaeger taxonomy database generation pipeline"""
    if model == "default":
        raise click.BadParameter(
            f"Model '{model}' is not supported"
        )

    else:
        from jaeger.commands.taxonomy import predict_taxonomy
        predict_taxonomy(**kwargs)

main.add_command(test)
main.add_command(predict)
main.add_command(train)
main.add_command(register_models)
main.add_command(download)
main.add_command(utils)
main.add_command(taxonomy)


if __name__ == "__main__":
    main()
