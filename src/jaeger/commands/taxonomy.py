# experimental taxonomy predicton workflow
import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import traceback
import sys
import psutil
import shutil
import time

try:
    import taxopy
    import faiss
except ModuleNotFoundError as exc:
    raise ImportError(
        "The taxonomy workflow requires the 'taxopy' and 'faiss' packages. "
        "Install them with e.g.: pip install taxopy faiss-cpu"
    ) from exc

import numpy as np
import tensorflow as tf
from typing import Union
from pathlib import Path
from importlib.resources import files
from importlib.metadata import version
from typing import Any, List
from jaeger.nnlib.inference import InferModel
from jaeger.seqops.io import fragment_generator
from jaeger.utils.gpu import get_device_name
from jaeger.utils.fs import validate_fasta_entries
from jaeger.utils.misc import json_to_dict, AvailableModels, get_model_id  # noqa: F401
from jaeger.utils.logging import description, get_logger


# Standard abbreviated rank prefixes for the full-lineage string.
_RANK_PREFIXES = {
    "superkingdom": "d",
    "kingdom": "k",
    "subkingdom": "sk",
    "superphylum": "sp",
    "phylum": "p",
    "subphylum": "sph",
    "superclass": "sc",
    "class": "c",
    "subclass": "ssc",
    "infraclass": "ic",
    "superorder": "so",
    "order": "o",
    "suborder": "sor",
    "infraorder": "io",
    "parvorder": "po",
    "superfamily": "sf",
    "family": "f",
    "subfamily": "sfa",
    "tribe": "t",
    "subtribe": "st",
    "genus": "g",
    "subgenus": "sg",
    "species group": "sg",
    "species subgroup": "ssg",
    "species": "s",
    "subspecies": "ss",
    "forma": "fo",
    "varietas": "v",
    "biotype": "b",
    "realm": "r",
    "subrealm": "sr",
}


def _format_ranked_lineage(lca) -> str:
    """Return a `r__Name;k__Name;...` lineage string from a taxopy Taxon."""
    parts = []
    for rank, name in getattr(lca, "rank_name_dictionary", {}).items():
        prefix = _RANK_PREFIXES.get(rank, rank[0] if rank else "?")
        parts.append(f"{prefix}__{name}")
    return ";".join(parts)


class TaxonomyModel:
    def __init__(
        self,
        faiss_path: Union[str, Path],
        index2taxid_path: Union[str, Path],
        taxdump_path: Union[str, Path],
    ):
        self.index2taxid = self.load_index2taxid(path=index2taxid_path)
        self.faiss_index = self.load_faiss_index(path=faiss_path)
        self.taxdb = self.load_taxdb(path=taxdump_path)
        # Cache Taxon objects because constructing them is expensive.
        self._taxon_cache: dict[int, Any] = {}

    def load_taxdb(self, path: Union[str, Path]):
        return taxopy.TaxDb(
            nodes_dmp=path / "nodes.dmp",
            names_dmp=path / "names.dmp",
            merged_dmp=path / "merged.dmp",
        )

    def load_index2taxid(self, path):
        data = np.load(path)
        return data["taxids"]

    def load_faiss_index(self, path):
        return faiss.read_index(str(path))

    def _get_taxon(self, taxid: int):
        taxon = self._taxon_cache.get(taxid)
        if taxon is None:
            taxon = taxopy.Taxon(taxid, self.taxdb)
            self._taxon_cache[taxid] = taxon
        return taxon

    def get_lca(self, taxids: List[int], fraction=0.6):
        taxons = [self._get_taxon(i) for i in taxids]
        if len(taxons) > 1:
            return taxopy.find_majority_vote(
                taxons, taxdb=self.taxdb, fraction=fraction
            )
        return taxons[0]

    def predict(self, embeddings: List, headers: List, k: int = 1):
        """Batch all contig windows into a single FAISS search.

        ``embeddings`` is a list of 2-D arrays, one per contig. The result is an
        LCA call per contig using the ``k`` nearest neighbors of every window.
        """
        if not embeddings:
            return []

        # Stack all windows from all contigs into one contiguous float32 array.
        all_embeddings = np.vstack(embeddings).astype(np.float32, copy=False)
        all_embeddings = np.ascontiguousarray(all_embeddings)
        faiss.normalize_L2(all_embeddings)

        _, all_indices = self.faiss_index.search(all_embeddings, k)
        all_taxids = self.index2taxid[all_indices.flatten()]

        # Map each window back to its contig.
        offsets = np.cumsum([0] + [e.shape[0] for e in embeddings]) * k

        tmp_l = []
        for idx, (embedding, header) in enumerate(zip(embeddings, headers)):
            start = offsets[idx]
            end = offsets[idx + 1]
            taxids = all_taxids[start:end]
            tmp_l.append({"header": header, "lca": self.get_lca(taxids)})
        return tmp_l


def copy_tax_files(in_tax, output_tax):
    src = Path(in_tax)
    dst = Path(output_tax)

    dst.mkdir(parents=True, exist_ok=True)  # Ensure destination exists

    for fname in ["names.dmp", "nodes.dmp", "merged.dmp"]:
        shutil.copy(src / fname, dst / fname)


def unravel(y_pred: dict, **kwargs) -> tuple[Any, Any, Any]:
    """
    Processes model predictions and associated metadata into structured dictionaries.
    num_classes
    fsize

    Returns:
        data (dict): Core prediction metrics and metadata
        data_full (dict): Full output for auxiliary processing
    """

    # -- Step 1: Determine split points
    # split_flags = np.array(y_pred["meta_2"], dtype=np.int32)
    # split_indices = np.where(split_flags == 1)[0] + 1

    # if y_pred["prediction"].shape[0] == split_indices[-1]:
    #     split_indices = split_indices[:-1]

    # -- Step 2: Split predictions and embeddings
    # _ = np.split(y_pred["prediction"], split_indices, axis=0)
    # _ = np.split(y_pred["reliability"], split_indices, axis=0)
    # embeddings = np.split(y_pred["embedding"], split_indices, axis=0)

    # -- headers
    # headers = np.split(y_pred["meta_0"], split_indices, axis=0)
    # return (embeddings, headers)

    taxids = [kwargs.get("acc2taxid").get(i.decode()) for i in y_pred["meta_0"]]

    return (y_pred["embedding"], taxids, y_pred["meta_0"])


def unravel_inference(y_pred: dict) -> tuple[Any, Any]:
    """
    Processes model predictions and associated metadata into tuple.

    Returns:
        embeddings
        headers
    """

    # -- Step 1: Determine split points
    split_flags = np.array(y_pred["meta_2"], dtype=np.int32)
    split_indices = np.where(split_flags == 1)[0] + 1

    if y_pred["prediction"].shape[0] == split_indices[-1]:
        split_indices = split_indices[:-1]

    # -- Step 2: Split predictions and embeddings
    _ = np.split(y_pred["prediction"], split_indices, axis=0)
    _ = np.split(y_pred["reliability"], split_indices, axis=0)
    embeddings = np.split(y_pred["embedding"], split_indices, axis=0)

    # -- headers
    headers = [i[0] for i in np.split(y_pred["meta_0"], split_indices, axis=0)]
    return (embeddings, headers)


def create_cosine_index(vectors: np.ndarray, save_path: str) -> None:
    """
    Creates a cosine similarity FAISS index and saves it to disk.

    Args:
        vectors (np.ndarray): 2D array of shape (n_samples, dim). Will be cast
            to float32 if necessary.
        save_path (str): Path to save the FAISS index (e.g., 'index.faiss').
    """
    assert vectors.ndim == 2, "Input must be a 2D array"
    vectors = vectors.astype(np.float32, copy=False)

    faiss.normalize_L2(vectors)  # Normalize vectors for cosine similarity

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(
        dim
    )  # Inner product = cosine similarity (after normalization)

    index.add(vectors)  # Add vectors to index
    faiss.write_index(index, str(save_path))


def get_acc2taxid(path: str):
    map_ = {}
    with open(path, "r") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if index == 0 or not line:
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            map_[fields[1]] = int(fields[2])
    return map_


def build_taxdb(**kwargs):
    # acc2tax, tax, input, out
    current_process = psutil.Process()
    GB_BYTES = 1024**3
    MODEL = kwargs.get("model")
    DATA_PATH = files("jaeger.data")
    if kwargs.get("config") is None:
        CONFIG_PATH = DATA_PATH / "config.json"
    else:
        CONFIG_PATH = kwargs.get("config")

    USER_MODEL_PATHS = json_to_dict(CONFIG_PATH).get("model_paths")
    MODEL_INFO = AvailableModels(path=USER_MODEL_PATHS).info[MODEL]
    MEMORY_LIMIT = 1024 * kwargs.get("mem", 4)
    THREADS = kwargs.get("workers")
    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem

    OUTPUT_DIR = Path(kwargs.get("output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    acc2taxid = get_acc2taxid(kwargs.get("acc2tax"))
    log_file = Path(f"{file_base}_jaeger.log")
    logger = get_logger(OUTPUT_DIR, log_file, level=kwargs.get("verbose"))
    logger.info(
        description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters")
    )
    logger.debug(DATA_PATH)
    logger.debug(AvailableModels(path=USER_MODEL_PATHS).info)
    logger.debug(MODEL_INFO)

    try:
        num = validate_fasta_entries(str(input_file_path), min_len=kwargs.get("fsize"))
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_faissdb = OUTPUT_DIR / "genomes.faiss"
    output_tax = OUTPUT_DIR / "tax"
    output_tax.mkdir(exist_ok=True)
    output_index2tax = OUTPUT_DIR / "index2tax.npz"

    if output_faissdb.exists() and not kwargs.get("overwrite"):
        logger.error(
            "a faiss index exists in the output path. enable --overwrite option to overwrite the output file."
        )
        sys.exit(1)

    if not MODEL_INFO["graph"].exists():
        logger.error(f"could not find model graph. please check {USER_MODEL_PATHS}")
        sys.exit(1)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    except RuntimeError:
        # Thread counts can only be set before the TF runtime initializes;
        # keep the existing configuration in already-initialized processes.
        logger.warning(
            f"TensorFlow runtime already initialized; using default threading (requested {THREADS})"
        )
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    mode = None

    if kwargs.get("cpu"):
        mode = "CPU"
        tf.config.set_visible_devices([], "GPU")
        logger.info("CPU only mode selected")
    elif gpus:
        mode = "GPU"
        tf.config.set_visible_devices([gpus[kwargs.get("physicalid")]], "GPU")

        # Set mixed precision policy if requested
        precision = kwargs.get("precision", "fp32")
        if precision == "fp16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("GPU precision: mixed_float16 (FP16 compute, FP32 variables)")
        elif precision == "bf16":
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            logger.info("GPU precision: mixed_bfloat16 (BF16 compute, FP32 variables)")
        else:
            logger.info("GPU precision: float32 (FP32)")

        try:
            tf.config.set_logical_device_configuration(
                gpus[kwargs.get("physicalid")],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=MEMORY_LIMIT, experimental_device_ordinal=10
                    )
                ],
            )
        except Exception as e:
            logger.error(f"an error {e} occurred during virtual device initialization ")
            logger.debug(traceback.format_exc())
    else:
        mode = "CPU"
        logger.warning(
            "could not find a GPU on the system. For optimal performance run Jaeger on a GPU."
        )

    logger.info(f"tensorflow: {version('tensorflow')}")
    logger.info(f"input file: {input_file}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {OUTPUT_DIR.resolve()}")
    logger.info(f"fragment size: {kwargs.get('fsize')}")
    logger.info(f"stride: {kwargs.get('stride')}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"avail mem: {psutil.virtual_memory().available / (GB_BYTES):.2f}GB")
    logger.info(
        f"intra threads: {tf.config.threading.get_intra_op_parallelism_threads()}"
    )
    logger.info(
        f"inter threads: {tf.config.threading.get_inter_op_parallelism_threads()}"
    )
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(
        f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)"
    )

    device = tf.config.list_logical_devices(mode)
    device_names = [get_device_name(i) for i in device]
    logger.debug(f"{device}, {device_names}")
    if len(device) > 1:
        logger.info(f"Using MirroredStrategy {device_names}")
        strategy = tf.distribute.MirroredStrategy(device_names)
    else:
        logger.info(f"Using OneDeviceStrategy {device_names}")
        strategy = tf.distribute.OneDeviceStrategy(device_names[0])

    use_xla = kwargs.get("xla", False)
    if use_xla:
        logger.info(
            "Using XLA-compiled inference (first batch may be slow due to compilation)"
        )
    try:
        logger.info("loading model")
        model = InferModel(MODEL_INFO, use_xla=use_xla, return_embedding=True)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    string_processor_config = model.string_processor_config
    logger.info("generating sequence fragments")
    input_dataset = tf.data.Dataset.from_generator(
        lambda: fragment_generator(
            str(input_file_path),
            no_progress=False,
            fragsize=kwargs.get("fsize"),
            stride=kwargs.get("stride"),
            num=num,
        ),
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
    )

    from jaeger.seqops.encode import process_string_inference

    logger.info("preprocessing fragments")
    idataset = (
        input_dataset.map(
            process_string_inference(
                crop_size=kwargs.get("fsize"),
                codons=string_processor_config.get("codon"),
                codon_num=string_processor_config.get("codon_id"),
                codon_depth=string_processor_config.get("codon_depth"),
                ngram_width=string_processor_config.get("ngram_width"),
                seq_onehot=string_processor_config.get("seq_onehot"),
                input_type=string_processor_config.get("input_type"),
                masking=string_processor_config.get("masking"),
                mutate=string_processor_config.get("mutate"),
                mutation_rate=string_processor_config.get("mutation_rate"),
                shuffle=string_processor_config.get("shuffle"),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(kwargs.get("batch"), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(25)
    )

    with strategy.scope():
        try:
            logger.info("running embedding inference")
            y_pred = model.predict(idataset, no_progress=True)
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(
                f"an error {e} occured during generating embeddings on {'|'.join(device_names)}! check {log_file} for traceback."
            )
            sys.exit(1)

    # save embeddings and headers
    logger.info("saving embeddings")
    embeddings, taxids, _ = unravel(y_pred, acc2taxid=acc2taxid)
    np.savez(OUTPUT_DIR / "embeddings.npz", embeddings=embeddings, taxids=taxids)
    np.savez(output_index2tax, taxids=taxids)
    # build faiss db
    logger.info("building FAISS index")
    create_cosine_index(embeddings, save_path=output_faissdb)
    # cp taxdump
    logger.info("copying taxdump files")
    copy_tax_files(in_tax=kwargs.get("tax"), output_tax=output_tax)


def predict_taxonomy(**kwargs):
    # acc2tax, tax, input, out
    current_process = psutil.Process()
    GB_BYTES = 1024**3
    MODEL = kwargs.get("model")
    DATA_PATH = files("jaeger.data")
    if kwargs.get("config") is None:
        CONFIG_PATH = DATA_PATH / "config.json"
    else:
        CONFIG_PATH = kwargs.get("config")

    USER_MODEL_PATHS = json_to_dict(CONFIG_PATH).get("model_paths")
    MODEL_INFO = AvailableModels(path=USER_MODEL_PATHS).info[MODEL]
    MEMORY_LIMIT = 1024 * kwargs.get("mem", 4)
    THREADS = kwargs.get("workers")
    input_file_path = Path(kwargs.get("input"))
    input_file = input_file_path.name
    file_base = input_file_path.stem

    OUTPUT_DIR = Path(kwargs.get("output"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR = Path(kwargs.get("db"))

    log_file = Path(f"{file_base}_jaeger.log")
    logger = get_logger(OUTPUT_DIR, log_file, level=kwargs.get("verbose"))
    logger.info(
        description(version("jaeger-bio")) + "\n{:-^80}".format("validating parameters")
    )
    logger.debug(DATA_PATH)
    logger.debug(AvailableModels(path=USER_MODEL_PATHS).info)
    logger.debug(MODEL_INFO)

    # load taxdb
    logger.info("loading taxonomy database")
    tax_model = TaxonomyModel(
        faiss_path=DATABASE_DIR / "genomes.faiss",
        taxdump_path=DATABASE_DIR / "tax",
        index2taxid_path=DATABASE_DIR / "index2tax.npz",
    )

    try:
        num = validate_fasta_entries(str(input_file_path), min_len=kwargs.get("fsize"))
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    output_taxonomy = OUTPUT_DIR / f"{file_base}_taxonomy.tsv"

    if output_taxonomy.exists() and not kwargs.get("overwrite"):
        logger.error(
            "output path is not empty. enable --overwrite option to overwrite the output file."
        )
        sys.exit(1)

    if not MODEL_INFO["graph"].exists():
        logger.error(f"could not find model graph. please check {USER_MODEL_PATHS}")
        sys.exit(1)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    except RuntimeError:
        # Thread counts can only be set before the TF runtime initializes;
        # keep the existing configuration in already-initialized processes.
        logger.warning(
            f"TensorFlow runtime already initialized; using default threading (requested {THREADS})"
        )
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    mode = None

    if kwargs.get("cpu"):
        mode = "CPU"
        tf.config.set_visible_devices([], "GPU")
        logger.info("CPU only mode selected")
    elif gpus:
        mode = "GPU"
        tf.config.set_visible_devices([gpus[kwargs.get("physicalid")]], "GPU")

        # Set mixed precision policy if requested
        precision = kwargs.get("precision", "fp32")
        if precision == "fp16":
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("GPU precision: mixed_float16 (FP16 compute, FP32 variables)")
        elif precision == "bf16":
            tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
            logger.info("GPU precision: mixed_bfloat16 (BF16 compute, FP32 variables)")
        else:
            logger.info("GPU precision: float32 (FP32)")

        try:
            tf.config.set_logical_device_configuration(
                gpus[kwargs.get("physicalid")],
                [
                    tf.config.LogicalDeviceConfiguration(
                        memory_limit=MEMORY_LIMIT, experimental_device_ordinal=10
                    )
                ],
            )
        except Exception as e:
            logger.error(f"an error {e} occurred during virtual device initialization ")
            logger.debug(traceback.format_exc())
    else:
        mode = "CPU"
        logger.warning(
            "could not find a GPU on the system. For optimal performance run Jaeger on a GPU."
        )

    logger.info(f"tensorflow: {version('tensorflow')}")
    logger.info(f"input file: {input_file}")
    logger.info(f"log file: {log_file.name}")
    logger.info(f"outpath: {OUTPUT_DIR.resolve()}")
    logger.info(f"fragment size: {kwargs.get('fsize')}")
    logger.info(f"stride: {kwargs.get('stride')}")
    logger.info(f"batch size: {kwargs.get('batch')}")
    logger.info(f"mode: {mode}")
    logger.info(f"avail mem: {psutil.virtual_memory().available / (GB_BYTES):.2f}GB")
    logger.info(
        f"intra threads: {tf.config.threading.get_intra_op_parallelism_threads()}"
    )
    logger.info(
        f"inter threads: {tf.config.threading.get_inter_op_parallelism_threads()}"
    )
    logger.info(f"CPU time(s) : {current_process.cpu_times().user:.2f}")
    logger.info(f"wall time(s) : {time.time() - current_process.create_time():.2f}")
    logger.info(
        f"memory usage : {current_process.memory_full_info().rss / GB_BYTES:.2f}GB ({current_process.memory_percent():.2f}%)"
    )

    device = tf.config.list_logical_devices(mode)
    device_names = [get_device_name(i) for i in device]
    logger.debug(f"{device}, {device_names}")
    if len(device) > 1:
        logger.info(f"Using MirroredStrategy {device_names}")
        strategy = tf.distribute.MirroredStrategy(device_names)
    else:
        logger.info(f"Using OneDeviceStrategy {device_names}")
        strategy = tf.distribute.OneDeviceStrategy(device_names[0])

    use_xla = kwargs.get("xla", False)
    if use_xla:
        logger.info(
            "Using XLA-compiled inference (first batch may be slow due to compilation)"
        )
    try:
        logger.info("loading model")
        model = InferModel(MODEL_INFO, use_xla=use_xla, return_embedding=True)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    string_processor_config = model.string_processor_config
    logger.info("generating sequence fragments")
    input_dataset = tf.data.Dataset.from_generator(
        lambda: fragment_generator(
            str(input_file_path),
            no_progress=False,
            fragsize=kwargs.get("fsize"),
            stride=kwargs.get("stride"),
            num=num,
        ),
        output_signature=(tf.TensorSpec(shape=(), dtype=tf.string)),
    )

    from jaeger.seqops.encode import process_string_inference

    logger.info("preprocessing fragments")
    idataset = (
        input_dataset.map(
            process_string_inference(
                crop_size=kwargs.get("fsize"),
                codons=string_processor_config.get("codon"),
                codon_num=string_processor_config.get("codon_id"),
                codon_depth=string_processor_config.get("codon_depth"),
                ngram_width=string_processor_config.get("ngram_width"),
                seq_onehot=string_processor_config.get("seq_onehot"),
                input_type=string_processor_config.get("input_type"),
                masking=string_processor_config.get("masking"),
                mutate=string_processor_config.get("mutate"),
                mutation_rate=string_processor_config.get("mutation_rate"),
                shuffle=string_processor_config.get("shuffle"),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(kwargs.get("batch"), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(25)
    )

    with strategy.scope():
        try:
            logger.info("running embedding inference")
            y_pred = model.predict(idataset, no_progress=True)
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(
                f"an error {e} occured during generating embeddings on {'|'.join(device_names)}! check {log_file} for traceback."
            )
            sys.exit(1)

    # unravel embedding vectors
    logger.info("running taxonomy assignment")
    embeddings, headers = unravel_inference(y_pred)

    predictions = tax_model.predict(embeddings, headers)

    # Write predictions to TSV using pandas for consistency with the rest of the package.
    logger.info("writing taxonomy predictions")
    import pandas as pd

    rows = []
    for pred in predictions:
        header = pred["header"]
        if isinstance(header, bytes):
            header = header.decode("utf-8")
        lca = pred["lca"]
        rows.append(
            {
                "header": header,
                "taxid": getattr(lca, "taxid", ""),
                "name": getattr(lca, "name", ""),
                "rank": getattr(lca, "rank", ""),
                "lineage": _format_ranked_lineage(lca),
            }
        )

    pd.DataFrame(rows).to_csv(output_taxonomy, sep="\t", index=False)
    logger.info(f"taxonomy predictions written to {output_taxonomy}")
