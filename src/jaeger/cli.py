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


@click.command(context_settings={'show_default': True})
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
    default=2000,
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=2000,
    help="The gap between two the sliding windows (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--model_path",
    default=None,
    help=(
        "Give the path to a model. overrides --model"
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
    help="Extract and report prophage-like regions",
)
@click.option(
    "-s",
    "--sensitivity",
    type=float,
    default=1.5,
    help="Sensitivity of the prophage extraction algorithm (0-4)",
)
@click.option(
    "--lc",
    type=int,
    default=500_000,
    help="Minimum contig length for prophage extraction",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--pc",
    type=int,
    default=3,
    help="Minimum phage score required to accept predictions",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use"
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
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "--getalllabels",
    is_flag=True,
    help="Get predicted labels for Non-Viral contigs",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
    default=1,
)
@click.option("-f", "--overwrite", is_flag=True, help="Overwrite existing files")
def predict(**kwargs):
    """
    Runs Jaeger on a dataset
    """
    model = kwargs.get("model")
    model_path = kwargs.get("model_path")
    config = kwargs.get("config")

    if model_path:
        model = "from_path"
    else:
        if model is None:
            model = "default"
        elif config is None and model not in AVAIL_MODELS:
            raise click.BadParameter(
                f"Model '{model}' is not one of the available options: {', '.join(AVAIL_MODELS)}"
            )

    """Run jaeger inference pipeline."""
    if model == "default":
        from jaeger.commands.predict_legacy import run_core

        run_core(**kwargs)
    else:
        from jaeger.commands.predict import run_core

        run_core(**kwargs)


@click.command(context_settings={'show_default': True})
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


@click.command(context_settings={'show_default': True})
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
    "--force",
    is_flag=True,
    required=False,
    help="delete existing checkpoints and continue",
)
@click.option(
    "--only_save",
    is_flag=True,
    required=False,
    help="save the model with weights from last checkpoints without training",
)
@click.option(
    "--mixed_precision",
    is_flag=True,
    required=False,
    help="use mix-precision floats to speed up training",
)
@click.option(
    "--meta",
    type=click.Path(
        exists=False,
        file_okay=True,
    ),
    required=False,
    help="Path to write metadata from the container",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
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
            f"{', '.join('--' + flag for flag in selected)} are mutually exclusive. Please specify only one."
        )
    
    if kwargs.get("from_last_checkpoint") and kwargs.get("force"):
        raise click.UsageError(
            "--from_last_checkpoint and --force are mutually exclusive. Please specify only one."
        )

    from jaeger.commands.train import train_fragment_core

    train_fragment_core(**kwargs)


@click.command(context_settings={'show_default': True})
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
    help="Path to jager config file (usefule when using Apptainer or Docker)",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
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


@click.command(context_settings={'show_default': True})
@click.option(
    "-p",
    "--path",
    type=click.Path(file_okay=False),
    required=False,
    default=None,
    help="Path to save model weights, graph and configuration files",
)
@click.option(
    "-c",
    "--config",
    type=str,
    required=False,
    help="Path to jager config file (useful when using Apptainer or Docker)",
)
@click.option(
    "-m", "--model_name",
    required=False,
    default=None,
    help="identifier of the model to download"
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
    help="Verbosity level: -vv debug, -v info",
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
    context_settings=dict(ignore_unknown_options=True, show_default=True),
    help="""
            Shuffles/generates random complexity input DNA sequences in .fasta or 
            .txt files to generate train (validation or train) sets for training (testing)
            out-of-distribution detection models (reliability models)
            if -ip is provided, wrong prediction are also added to the negative class
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
@click.option(
    "-ip",
    "--input_predictions",
    type=click.Path(exists=True),
    required=False,
    help="Path to jaeger predictions of the input (optional)",
)
@click.option("-o", 
              "--output", 
              type=str, 
              required=True, 
              help="Path to output file")
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

@click.option(
    "--num_tandem_repeats",
    type=int,
    default=0,
    required=False,
    help="generate n random tandem repeats",
)

@click.option(
    "--class_col",
    type=int,
    default=None,
    required=False,
    help="csv col with class id (when --itype CSV)",
)

@click.option(
    "--seq_col",
    type=int, 
    default=None,
    required=False,
    help="csv col with sequence (when --itype CSV)",
)
def ood_data(**kwargs):
    from jaeger.commands.utils import shuffle_core
    if kwargs.get("itype") == "CSV":
        seq_col = kwargs.get("seq_col")
        class_col = kwargs.get("class_col")

        if not (seq_col is not  None and class_col is not None):
            raise click.UsageError(
                "When --itype CSV is used, both --seq-col and --class-col must be provided."
        )

    shuffle_core(**kwargs)


@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
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
    context_settings=dict(ignore_unknown_options=True, show_default=True),
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
    help="min percentage of positions to mask (0.0)",
    default=0.0,
)
@click.option(
    "--maxperc",
    type=float,
    required=False,
    help="max percentage of positions to mask (1.0)",
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

    if kwargs.get('maxperc') < kwargs.get('minperc'):
        raise click.BadParameter("maxperc can not be lower than minperc")

    mask_core(**kwargs)


@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
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
@click.option("--class", type=int, required=False, help="class label")
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
    "--intype",
    type=click.Choice(["CSV", "FASTA"], 
    case_sensitive=False),
    required=True,
    help="input type",
    default="CSV",
)
@click.option(
    "--class_col",
    type=int,
    required=False,
    help="csv col with class id",
)
@click.option(
    "--seq_col",
    type=int, 
    required=True,
    help="csv col with sequence",
)
@click.option(
    "--outtype",
    type=click.Choice(["CSV", "FASTA"], case_sensitive=False),
    required=True,
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
def dataset(**kwargs):
    """Generate a non-redundant fragment database from fasta file for training/validating
    fragment models"""
    from jaeger.commands.utils import dataset_core

    intype = kwargs.get("intype", "").upper()

    match intype:
        case "CSV":
            if not (kwargs.get("class_col") is not None and kwargs.get("seq_col") is not None):
                raise click.UsageError("For CSV input, you must specify both --seq_col and --class_col.")
        case "FASTA":
            if kwargs.get("class") is None:
                raise click.UsageError("For FASTA input, you must specify --class.")
        case _:
            raise click.UsageError("Invalid input type: must be 'CSV' or 'FASTA'.")

    dataset_core(**kwargs)
    pass

@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
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

@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
    help="""Combine multiple model graphs into an ensemble model. Outputs can be summarized with majority voting,
            sum or mean of logits of each model

            usage\n
            -----

            jaeger utils combine_models -i path/to/model1 -i path/to/model2 -i path/to/model3 -c mv
        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to saved model",
    multiple=True
)
@click.option("-o", "--output", type=str, required=True, help="Path to save the ensemble model")
@click.option(
    "-c",
    "--comb",
    type=click.Choice(["MV", "SUM", "MEAN", "NONE"], case_sensitive=False),
    required=True,
    help="how to summarize the model outputs " \
    "MV: majority voting" \
    "SUM: sum of logits" \
    "MEAN: Mean of logits" \
    "NONE: Do not aggregate",
)
def combine_models(**kwargs):

    from jaeger.commands.utils_models import combine_models_core

    combine_models_core(**kwargs)

@utils.command(
    context_settings=dict(ignore_unknown_options=True, show_default=True),
    help="""
            Calculate stats from Jaeger output

            usage                                                                      
            -----                                                                      

            jaeger utils stats -i jaeger_output.tsv -o jaeger_output_stats
            
        """,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(exists=True),
    required=True,
    help="Path to input .tsv file",
)
@click.option("-o", "--output", type=str, required=True, help="Path to output dir")

def stats(**kwargs):
    """Calculate stats from Jaeger output"""
    from jaeger.commands.utils import stats_core

    stats_core(**kwargs)
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
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="The length of the gap between two sliding windows (stride==fsize)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available choices (when --config is not set): {', '.join(AVAIL_MODELS)}"
    ),
)
@click.option(
    "--model_path",
    default=None,
    help=(
        "Give the path to a model. overrides --model"
    ),
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to Jaeger config file (useful when using Apptainer or Docker)",
)
@click.option(
    "--rc",
    type=float,
    default=0.1,
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use"
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
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
    help="Length of the sliding window",
)
@click.option(
    "--stride",
    type=int,
    default=2048,
    help="The gap between two sliding of the sliding window (fsize = stride)",
)
@click.option(
    "-m",
    "--model",
    default="default",
    help=(
        f"Select a deep-learning model to use. "
        f"Available models: (when --config is not set): {', '.join(AVAIL_MODELS)}"
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
    help="Minimum reliability score required to accept predictions",
)
@click.option(
    "--batch",
    type=int,
    default=96,
    help="Parallel batch size, lower if GPU runs out of memory",
)
@click.option(
    "--workers", type=int, default=4, help="Number of threads to use"
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Ignore available GPUs and explicitly run on CPU",
)
@click.option(
    "--physicalid",
    type=int,
    default=0,
    help="Set default GPU device ID for multi-GPU systems",
)
@click.option(
    "--mem",
    type=int,
    default=4,
    help="GPU memory limit",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbosity level: -vv debug, -v info",
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
