import numpy as np
from typing import Any, Dict
import pandas as pd
import traceback
import pyfastx
import logging
from jaeger.postprocess.helpers import (get_window_summary,
                                 update_dict,
                                 ood_predict_default,
                                 softmax_entropy)

logger = logging.getLogger("jaeger")

def pred_to_dict(config, y_pred, **kwargs) -> pd.DataFrame:
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
            np.array(y_pred["meta"][0], dtype=np.unicode_), split_indices, axis=0
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


def generate_summary(config, data) -> pd.DataFrame:
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


def write_output(config:Any,
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

    try:
        df = generate_summary(config, data)
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
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())


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


