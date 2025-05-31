import numpy as np
from typing import Any, Dict
import pandas as pd
import traceback
import pyfastx
import logging
from jaeger.postprocess.helpers import (get_window_summary,
                                 update_dict,
                                 ood_predict_default,
                                 sigmoid,
                                 softmax_entropy)

logger = logging.getLogger("jaeger")

# legacy methods
def pred_to_dict_legacy(config, y_pred, **kwargs) -> pd.DataFrame:
            # output preprocessing
        split_indices = (
            np.where(np.array(y_pred["meta"][2], dtype=np.int32) == 1)[0] + 1
        )

        if y_pred["y_hat"]["output"].shape[0] == split_indices[-1]:
            split_indices = split_indices[:-1]
        predictions = np.split(y_pred["y_hat"]["output"], split_indices, axis=0)

        ood = np.split(
            y_pred["y_hat"]["embedding"], split_indices, axis=0
        )  # get params
        ood = list(map(lambda x: ood_predict_default(x, kwargs.get('ood_params'))[0], ood))

        headers = np.split(
            np.array(y_pred["meta"][0], dtype=np.str_), split_indices, axis=0
        )
        lengths = np.split(
            np.array(y_pred["meta"][4], dtype=np.int32), split_indices, axis=0
        )
        gc_skews = np.split(y_pred["meta"][-1].astype(float), split_indices, axis=0)
        g = y_pred["meta"][-4].astype(float)
        c = y_pred["meta"][-5].astype(float)
        a = y_pred["meta"][-3].astype(float)
        t = y_pred["meta"][-2].astype(float)
        ns = (kwargs.get("fsize") - (a + t + g + c)) / kwargs.get("fsize")
        ns = np.split(ns, split_indices, axis=0)
        gcs = (g + c) / kwargs.get("fsize")
        gcs = np.split(gcs, split_indices, axis=0)

        lengths = np.array(list(map(lambda x: x[0], lengths)))
        headers = np.array(list(map(lambda x: x[0], headers)))

        pred_sum = np.array(
            list(map(lambda x: np.mean(x, axis=0), predictions)), np.float16
        )
        pred_var = np.array(
            list(map(lambda x: np.var(x, axis=0), predictions)), np.float16
        )
        consensus = np.argmax(pred_sum, axis=1)
        frag_pred = list(map(lambda x: np.argmax(x, axis=-1), predictions))
        per_class_counts = list(
            map(lambda x: np.unique(x, return_counts=True), frag_pred)
        )
        per_class_counts = list(
            map(
                lambda x, n=config["num_classes"]: update_dict(x, n),
                per_class_counts,
            )
        )
        entropy_pred = list(map(lambda x: softmax_entropy(x), predictions))
        entropy_mean = np.array(
            list(map(lambda x: np.mean(x, axis=0), entropy_pred)), np.float16
        )
        prophage_contam = (pred_sum[:, 1] < pred_var[:, 1]) * (consensus == 0)
        host_contam = (pred_sum[:, 1] < pred_var[:, 1]) * (consensus == 1)

        data = {
            "headers": headers,
            "length": lengths,
            "consensus": consensus,
            "per_class_counts": per_class_counts,
            "pred_sum": pred_sum,
            "pred_var": pred_var,
            "frag_pred": frag_pred,
            "ood": ood,
            "entropy": entropy_mean,
            "host_contam": host_contam,
            "prophage_contam": prophage_contam,
            "repeats": kwargs.get('term_repeats'),
            "gc": gcs,
            "ns": ns,
        }
        data_full = {
            "predictions": predictions,
            "headers":headers,
            "lengths":lengths,
            "gc_skews":gc_skews,
            "gcs":gcs,
        }
        return data, data_full


def generate_summary_legacy(config, data) -> pd.DataFrame:
    """
    Generates a per contig summary based on the provided configuration,
    arguments, and data.

    Args:
    ----
        config (dict): The configuration settings.
        model (str): The selected model.
        data (dict): The data used to create the summary.
        output_file_path (Path): The output file path for the summary.
        output_phage_file_path (Path): The output file path for phage-related summary.

    Returns:
    -------
        None
    """

    logger.info("Generating summary")
    
    class_map = config["labels"]  # Comes from config
    lab = {int(k): v for k, v in config["all_labels"].items()}
    
    # Basic summary columns
    columns = {
        "contig_id": data["headers"],
        "length": data["length"],
        "prediction": [class_map[x] for x in data["consensus"]],
        "entropy": data["entropy"],
        "reliability_score": [np.mean(x) for x in data["ood"]],
        "host_contam": data["host_contam"],
        "prophage_contam": data["prophage_contam"],
    }

    # Handle additional default model-specific features
    if config["model"] == "default":
        columns["G+C"] = [np.mean(x) for x in data["gc"]]
        columns["N%"] = [np.mean(x) for x in data["ns"]]

        # Finding and appending the second highest class prediction
        ev = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([2, 1]), axis=1)
        av = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([3, 1]), axis=1) * 2
        bv = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([0, 1]), axis=1) * 3

        class_map2 = {int(k): v for k, v in config["second"].items()}
        columns["prediction_2"] = [class_map2[x] for x in (ev + av + bv)]

    # Appends class-wise information to the dictionary
    for i, label in lab.items():
        columns[f"#_{label}_windows"] = [x[i] for x in data["per_class_counts"]]
        columns[f"{label}_score"] = [x[i] for x in data["pred_sum"]]
        columns[f"{label}_var"] = [x[i] for x in data["pred_var"]]

    # Append the window summary column
    columns["window_summary"] = [
        get_window_summary(x, config["vindex"]) for x in data["frag_pred"]
    ]

    # Create dataframe and merge with repeat data
    df = pd.DataFrame(columns).set_index("contig_id")

    df = df.join(
        data["repeats"].set_index("contig_id")[["terminal_repeats", "repeat_length"]],
        how="right"
    ).reset_index(names="contig_id")

    # Replace "__" with "," in contig_id
    df["contig_id"] = df["contig_id"].str.replace("__", ",")

    return df


def write_output_legacy(config:Any,
                 data: Dict,
                 reliability_cutoff:float=0.5,
                 **kwargs):
    """
    Writes the output based on the provided arguments, configuration, and data.

    Args:
    ----
        config: The configuration settings.
        data: The data to be used for generating the output.

    Returns:
    -------
        None
    """

    #try:
    df = generate_summary_legacy(config, data)
    # Save the full summary
    df.to_csv(kwargs.get('output_table_path'),
            sep="\t",
            index=False,
            float_format="%.3f")

    # Save only phage-related sequences
    df.query(f'(prediction == "phage") and (phage_score > 3) and (reliability_score > {reliability_cutoff})') \
        .to_csv(kwargs.get('output_phage_table_path'),
                sep="\t",
                index=False,
                float_format="%.3f")

    logger.info("Summary generation completed!")
    #except Exception as e:



def pred_to_dict(y_pred: dict, **kwargs) -> tuple[dict, dict]:
    """
    Processes model predictions and associated metadata into structured dictionaries.
    num_classes
    fsize

    Returns:
        data (dict): Core prediction metrics and metadata
        data_full (dict): Full output for auxiliary processing
    """

    # -- Step 1: Determine split points
    split_flags = np.array(y_pred["meta_2"], dtype=np.int32)
    split_indices = np.where(split_flags == 1)[0] + 1

    if y_pred["prediction"].shape[0] == split_indices[-1]:
        split_indices = split_indices[:-1]

    # -- Step 2: Split predictions and embeddings
    predictions = np.split(y_pred["prediction"], split_indices, axis=0)
    ood = np.split(y_pred["reliability"], split_indices, axis=0)

    # -- Step 4: Split metadata fields
    headers = np.array([h[0] for h in np.split(np.array(y_pred["meta_0"], dtype=str), split_indices)])
    lengths = np.array([l[0] for l in np.split(np.array(y_pred["meta_4"], dtype=np.int32), split_indices)])
    gc_skews = np.split(y_pred["meta_9"].astype(float), split_indices)

    # -- Step 5: Nucleotide content
    a, t, g, c = map(lambda i: y_pred[i].astype(float), ["meta_7", "meta_8", "meta_6", "meta_5"])
    fsize = kwargs["fsize"]
    ns = (fsize - (a + t + g + c)) / fsize
    gcs = (g + c) / fsize

    ns = np.split(ns, split_indices)
    gcs = np.split(gcs, split_indices)

    # -- Step 6: Summary statistics
    pred_sum = np.array([np.mean(p, axis=0) for p in predictions], dtype=np.float16)
    pred_var = np.array([np.var(p, axis=0) for p in predictions], dtype=np.float16)
    consensus = np.argmax(pred_sum, axis=1)

    frag_pred = [np.argmax(p, axis=-1) for p in predictions]
    per_class_counts = [
        update_dict(np.unique(fp, return_counts=True), kwargs.get("num_classes"))
        for fp in frag_pred
    ]

    entropy_pred = [softmax_entropy(p) for p in predictions]
    ood = [sigmoid(p) for p in ood]
    entropy_mean = np.array([np.mean(e, axis=0) for e in entropy_pred], dtype=np.float16)

    # -- Step 7: Contamination heuristics
    prophage_contam = (pred_sum[:, 1] < pred_var[:, 1]) & (consensus == 0)
    host_contam = (pred_sum[:, 1] < pred_var[:, 1]) & (consensus == 1)

    # -- Step 8: Build output dicts
    data = {
        "headers": headers,
        "length": lengths,
        "consensus": consensus,
        "per_class_counts": per_class_counts,
        "pred_sum": pred_sum,
        "pred_var": pred_var,
        "frag_pred": frag_pred,
        "ood": ood,
        "entropy": entropy_mean,
        "host_contam": host_contam,
        "prophage_contam": prophage_contam,
        "repeats": kwargs.get("term_repeats"),
        "gc": gcs,
        "ns": ns,
    }

    data_full = {
        "predictions": predictions,
        "headers": headers,
        "lengths": lengths,
        "gc_skews": gc_skews,
        "gcs": gcs,
    }

    return data, data_full


def generate_summary(data, **kwargs) -> pd.DataFrame:
    """
    Generates a per contig summary based on the provided configuration,
    arguments, and data.

    Args:
    ----
        data (dict): The data used to create the summary.
        kwargs
        ------
        labels
        indices

        output_file_path (Path): The output file path for the summary.
        output_phage_file_path (Path): The output file path for phage-related summary.

    Returns:
    -------
        None
    """

    logger.info("Generating summary")
    classes_ = kwargs.get("labels")  # Comes from config
    indices_ = kwargs.get("indices")
    class_map = {int(k): v for k, v in zip(indices_, classes_)}
    
    # Basic summary columns
    columns = {
        "contig_id": data["headers"],
        "length": data["length"],
        "prediction": [class_map[x] for x in data["consensus"]],
        "entropy": data["entropy"],
        "reliability_score": [np.mean(x) for x in data["ood"]],
        "host_contam": data["host_contam"],
        "prophage_contam": data["prophage_contam"],
    }

    # Handle additional default model-specific features

    columns["G+C"] = [np.mean(x) for x in data["gc"]]
    columns["N%"] = [np.mean(x) for x in data["ns"]]

    # # Finding and appending the second highest class prediction
    # ev = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([2, 1]), axis=1)
    # av = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([3, 1]), axis=1) * 2
    # bv = np.prod(np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([0, 1]), axis=1) * 3

    # class_map2 = {int(k): v for k, v in classes_.items()}
    # columns["prediction_2"] = [class_map2[x] for x in (ev + av + bv)]

    # Appends class-wise information to the dictionary
    for i, label in class_map.items():
        columns[f"#_{label}_windows"] = [x[i] for x in data["per_class_counts"]]
    for i, label in class_map.items():
        columns[f"{label}_score"] = [x[i] for x in data["pred_sum"]]
        columns[f"{label}_var"] = [x[i] for x in data["pred_var"]]

    # Append the window summary column
    columns["window_summary"] = [
        get_window_summary(x, classes_.index("phage")) for x in data["frag_pred"]
    ]

    # Create dataframe and merge with repeat data
    df = pd.DataFrame(columns).set_index("contig_id")

    df = df.join(
        data["repeats"].set_index("contig_id")[["terminal_repeats", "repeat_length"]],
        how="right"
    ).reset_index(names="contig_id")

    # Replace "__" with "," in contig_id
    df["contig_id"] = df["contig_id"].str.replace("__", ",")

    return df


def write_output(data: Dict,
                 reliability_cutoff:float=0.5,
                 **kwargs):
    """
    Writes the output based on the provided arguments, configuration, and data.

    Args:
    ----
        config: The configuration settings.
        data: The data to be used for generating the output.

    Returns:
    -------
        None
    """

    #try:
    df = generate_summary(data, **kwargs)
    # Save the full summary
    df.to_csv(kwargs.get('output_table_path'),
            sep="\t",
            index=False,
            float_format="%.3f")

    # Save only phage-related sequences
    df.query(f'(prediction == "phage") and (phage_score > 3) and (reliability_score > {reliability_cutoff})') \
        .to_csv(kwargs.get('output_phage_table_path'),
                sep="\t",
                index=False,
                float_format="%.3f")
    return len(df)
    # logger.info("Summary generation completed!")
    #except Exception as e:



def write_fasta_from_results(input_fasta:str,
                             output_tsv: str,
                             output_fasta:str,
                             width:int = 70) -> None:
    """
    Generates a .fasta file given the input fasta file and a .tsv with predictions

    Args:
    ----
        input_fasta: path to the input fasta file
        output_tsv: path to tsv file with predictions
        output_fasta: path to output fasta file
        width: fasta line width

    Returns:
    -------
        None
    """

    logger.info(f"generating fasta file {output_fasta}")
    phages = set(pd.read_table(output_tsv)['contig_id'].to_list())
    phage_fasta = open(output_fasta, "w")
    for record in pyfastx.Fasta(input_fasta, build_index=False):
        if record[0] in phages:
            phage_fasta.write(f">{record[0]}\n")
            for i in range(0, len(record), width):
                phage_fasta.write(f">{record[0][i:i+width]}\n")


