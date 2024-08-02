"""

Copyright (c) 2024 Yasas Wijesekara

"""
import os
import sys
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyfastx
import parasail
import numpy as np
import ruptures as rpt
import progressbar
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kneed import KneeLocator
from pycirclize import Circos
from jaegeraa.utils import safe_divide

progressbar.streams.wrap_stderr()
logger = logging.getLogger("Jaeger")


def find_runs(x):  # sourcery skip: extract-method
    # from https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    """Find runs of consecutive items in an array.

    Args
    ----
    x : list or np.array
        a list or a numpy array with integers

    Returns
    -------
    a tuple of lists (run_values, run_lengths)
    """
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    # find run starts
    loc_run_start = np.empty(n, dtype=bool)
    loc_run_start[0] = True
    np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]

    # find run values
    run_values = x[loc_run_start]

    # find run lengths
    run_lengths = np.diff(np.append(run_starts, n))

    # return run_values, run_starts, run_lengths
    return run_values, run_lengths


def get_window_summary(x, phage_pos):
    """
    returns string representation of window-wise predictions

    Args
    ----

    x : list or np.array
        list or numpy array with window-wise integer class labels

    phage_pos : int
        integer value representing Phage or Virus class

    Returns
    -------
    a string with Vs and ns. Vs represent virus or phage windows. ns
    represent cellular windows

    """
    items, run_length = find_runs(x == phage_pos)
    run_length = np.array(run_length, dtype=np.unicode_)
    tmp = np.empty(items.shape, dtype=np.unicode_)
    # print(phage_pos, items, run_length)
    tmp[items != phage_pos] = "n"
    tmp[items == phage_pos] = "V"
    x = np.char.add(run_length, tmp)
    return "".join(x)


def reverse_complement(dna_sequence):
    """
    Returns the reverse complement of a given DNA sequence.

    Args:
    ----
        dna_sequence (str): The input DNA sequence to compute the
                            reverse complement.

    Returns:
    -------
        str: The reverse complement of the input DNA sequence.
    """

    complement_dict = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
        "-": "-",
        "N": "N",
        "W": "W",
        "S": "S",
        "Y": "R",
        "R": "Y",
        "M": "K",
        "K": "M",
        "B": "V",
        "V": "B",
        "H": "D",
        "D": "H",
        "a": "T",
        "t": "A",
        "g": "C",
        "c": "G",
    }
    return "".join(complement_dict.get(base, "N") for base in
                   reversed(dna_sequence))


def update_dict(x, num_classes=4):
    # sourcery skip: remove-redundant-constructor-in-dict-union
    """
    Updates a dictionary with key-value pairs from input data.

    Args:
    ----
        x: Tuple containing keys and values to update the dictionary.
        num_classes (int, optional): Number of classes for initializing
                                     the dictionary keys. Defaults to 4.

    Returns:
    -------
        None: The dictionary is updated in place.
    """

    return {i: 0 for i in range(num_classes)} | dict(zip(x[0], x[1]))


def shanon_entropy(p):
    """
    Calculates the Shannon entropy (information gain) of a probability
    distribution.

    Args:
    ----
        p (array-like): The probability distribution as an array-like object.

    Returns:
    -------
        float: The Shannon entropy value calculated from the input probability
        distribution.
    """

    p = np.array(p)
    result = np.where(p > 0.0000000001, p, -10)
    p_log = np.log2(result, out=result, where=result > 0)
    return -np.sum(p * p_log, axis=-1)


def softmax_entropy(x):
    """
    Calculates the entropy of a softmax output distribution.

    Args:
    ----
        x (array-like): The softmax output distribution as an array-like
                        object.

    Returns:
    -------
        float: The entropy value calculated from the softmax output
               distribution.
    """

    ex = np.exp(x)
    return shanon_entropy(ex / np.sum(ex, axis=-1).reshape(-1, 1))


def smoothen_scores(x, w=5):
    """
    Smoothes the scores of different classes using a moving average.

    Args:
    ----
        x (array-like): The input array containing scores for different
                        classes.
        w (int, optional): The window size for the moving average.
                           Defaults to 5.

    Returns:
    -------
        array-like: The smoothed scores for each class after applying the
                    moving average.
    """
    return np.column_stack(
        [np.convolve(x[:, i], np.ones(w) / w, mode="same")
         for i in range(x.shape[1])]
    )


def ood_predict(x_features, params):
    """
    Predicts out-of-distribution (OOD) probabilities using logistic regression
    parameters.

    Args:
    ----
        x_features (array-like): The input features to predict OOD
                                 probabilities.
        params (dict): Dictionary containing logistic regression parameters.

    Returns:
    -------
        tuple: A tuple containing the predicted OOD probabilities and logits.
    """

    # Normalize x_features using NumPy's built-in functions
    x_features = (x_features - np.mean(x_features,
                                       axis=-1,
                                       keepdims=True)) / np.std(x_features,
                                                                axis=-1,
                                                                keepdims=True)
    logits = np.dot(x_features,
                    params["coeff"].flatten()) + params["intercept"]
    return (1 / (1 + np.exp(logits))).flatten(), logits


def normalize(x):
    """
    Normalizes the input array along axis 1 using mean and standard deviation.

    Args:
    ----
        x (array-like): The input array to be normalized.

    Returns:
    -------
        array-like: The normalized array after subtracting the mean and
                    dividing by the standard deviation along axis 1.
    """

    x_mean = x.mean(axis=1).reshape(-1, 1)
    x_std = x.std(axis=1).reshape(-1, 1)
    return (x - x_mean) / x_std


def normalize_with_batch_stats(x, mean, std):
    """
    Normalizes the input array using batch mean and standard deviation.

    Args:
    ----
        x (array-like): The input array to be normalized.
        mean (array-like): The batch mean values.
        std (array-like): The batch standard deviation values.

    Returns:
    -------
        array-like: The normalized array using the provided batch mean and
                    standard deviation.
    """

    return (x - mean) / std


def normalize_l2(x):
    """
    Normalizes the input array along axis 1 using L2 norm.

    Args:
    ----
        x (array-like): The input array to be normalized.

    Returns:
    -------
        array-like: The L2 normalized array along axis 1.
    """

    return x / np.linalg.norm(x, 2, axis=1).reshape(-1, 1)


def ood_predict_default(x_features, params):
    """
    Predicts out-of-distribution (OOD) probabilities using logistic regression
    or a saved sklearn model.

    Args:
    ----
        x_features (array-like): The input features to predict OOD
                                 probabilities.
        params (dict): Dictionary containing parameters for prediction.

    Returns:
    -------
        tuple: A tuple containing the predicted OOD probabilities and logits
        based on the specified method.
    """

    # use parameters extimated using sklearn
    if params["type"] == "params":
        # x_features = normalize_with_batch_stats(x_features,
        # params['batch_mean']),
        # params['batch_std']
        x_features = normalize(x_features)
        logits = (
            np.dot(x_features,
                   params["coeff"].reshape(-1, 1)) + params["intercept"]
        )
        return (1 / (1 + np.exp(-logits))).flatten(), logits
    # use a saved a sklearn model
    elif params["type"] == "sklearn":
        features_data = normalize_with_batch_stats(
            x_features, params["batch_mean"], params["batch_std"]
        )
        features_data_l2 = normalize_l2(features_data)

        return params["model"].predict_proba(features_data_l2)[:, 0], 0


def get_ood_probability(ood, threshold=0.5):
    """
    Calculates the out-of-distribution (OOD) summary for alll windows by
    calculating the percentage of windows below a user-defined threshold.

    Args:
    ----
        ood (array-like): The input array of OOD values.
        threshold (float) : OOD probability threshold

    Returns:
    -------
        str: The OOD probability rounded to 2 decimal places if ood is not
             None, otherwise returns "-".
    """

    return f"{sum((ood < threshold) * 1) / len(ood) :2f}" if\
        ood is not None else "-"


def write_output(args, config, data):
    """
    Writes the output based on the provided arguments, configuration, and data.

    Args:
    ----
        args: The arguments for writing the output.
        config: The configuration settings.
        data: The data to be used for generating the output.

    Returns:
    -------
        None
    """

    try:
        generate_summary(config, args, data)
    except Exception as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


def generate_summary(config, args, data):
    """
    Generates a per contig summary based on the provided configuration,
    arguments, and data.

    Args:
    ----
        config (dict): The configuration settings.
        args: The arguments for generating the summary.
        data (dict): The data used to create the summary.

    Returns:
    -------
        None
    """

    logger.info("generating summary")
    class_map = config["labels"]  # comes from config
    lab = {int(k): v for k, v in config[args.model]["all_labels"].items()}
    # consider adding other infomation such as GC content in the future
    columns = {
        "contig_id": data["headers"],
        "length": data["length"],
        "prediction": list(map(lambda x: class_map[x], data["consensus"])),
        "entropy": data["entropy"],
        "reliability_score": list(map(lambda x: np.mean(x), data["ood"])),
        "host_contam": data["host_contam"],
        "prophage_contam": data["prophage_contam"],
    }

    if args.model == "default":
        generate_summary_default(data, columns, config, args)

    for i, label in lab.items():
        # appends the number of class-wise windows to the dict
        # logger.debug(data["per_class_counts"])
        columns[f"#_{label}_windows"] = list(
            map(lambda x, index=i: x[index], data["per_class_counts"])
        )

    for i, label in lab.items():
        # appends the class-wise scores and score variance to the dict
        columns[f"{label}_score"] = list(
            map(lambda x, index=i: x[index], data["pred_sum"])
        )
        columns[f"{label}_var"] = list(
            map(lambda x, index=i: x[index], data["pred_var"])
        )
    # append the window_summary col to the dict
    columns["window_summary"] = list(
        map(
            lambda x, phage_pos=config[args.model]["vindex"]:
                get_window_summary(
                    x, phage_pos
                ),
            data["frag_pred"],
        )
    )

    df = pd.DataFrame(columns).set_index("contig_id")

    df = df.join(
        data["repeats"].set_index("contig_id")[["terminal_repeats",
                                                "repeat_length"]],
        how="right",
    ).reset_index(names="contig_id")
    df["contig_id"] = df["contig_id"].apply(lambda x: x.replace("__", ","))
    df.to_csv(args.output_file_path, sep="\t", index=None, float_format="%.3f")
    df.query(
        'prediction == "phage" and phage_score > 3 and reliability_score > 0.2'
    ).to_csv(args.output_phage_file_path,
             sep="\t",
             index=None,
             float_format="%.3f")


def generate_summary_default(data, columns, config, args):
    """
    Generates additional summary information based on the default model
    configuration.

    Args:
    ----
        data (dict): The data used for generating the summary.
        columns (dict): The existing columns in the summary.
        config (dict): The configuration settings.
        args: The arguments related to the summary generation.

    Returns:
    -------
        None
    """

    columns["G+C"] = list(map(lambda x: np.mean(x), data["gc"]))
    columns["N%"] = list(map(lambda x: np.mean(x), data["ns"]))
    # finds and appends the second highest class to the dict-> prediction_2
    ev = np.prod(
        np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([2, 1]),
        axis=1
    )
    av = (
        np.prod(
            np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([3, 1]),
            axis=1,
        )
        * 2
    )
    bv = (
        np.prod(
            np.argsort(data["pred_sum"], axis=1)[:, 2:4] == np.array([0, 1]),
            axis=1,
        )
        * 3
    )
    class_map2 = {
        int(k): v for k, v in config[args.model]["second"].items()
    }  # comes from config
    columns["prediction_2"] = list(
        map(lambda x: class_map2[x], ev + av + bv)
    )  # only for deafult mode


def write_fasta(args):
    """
    Generates a .fasta file with the identified phage sequences configuration,
    arguments, and data.

    Args:
    ----
        config (dict): The configuration settings.
        args: The arguments for generating the summary.

    Returns:
    -------
        None
    """

    logger.info("generating .fasta file")
    phages = set(pd.read_table(args.output_phage_file_path)['contig_id'].to_list())
    phage_fasta = open(args.output_fasta_file_path, "w")
    for record in pyfastx.Fasta(args.input, build_index=False):
        if record[0] in phages:
            phage_fasta.write(f">{record[0]}\n{record[1]}\n")


def consecutive(data, stepsize=1):
    """
    Splits an array into subarrays where elements are consecutive.

    Args:
    ----
        data (array-like): The input array to split into consecutive subarrays.
        stepsize (int, optional): The step size between consecutive elements.
                                  Defaults to 1.

    Returns:
    -------
        list: A list of subarrays where elements are consecutive.
    """

    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def merge_overlapping_ranges(intervals):
    """
    Merge overlapping ranges in a list of intervals.

    Args
    ----
        intervals: List of intervals, each represented as [start, end].

    Returns
    -------
        merged_intervals: List of merged intervals.
    """
    if len(intervals) == 0:
        return []

    # Sort the intervals based on the start value
    sorted(intervals, key=lambda x: x[0])

    merged_intervals = [intervals[0]]

    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged_intervals[-1]

        if current_start <= last_end:  # Overlapping intervals
            merged_intervals[-1][1] = max(last_end, current_end)
        else:  # Non-overlapping intervals
            merged_intervals.append([current_start, current_end])

    return merged_intervals


def check_middle_number(array):
    """
    Checks the middle number in a given array based on specific conditions.

    Args:
    ----
        array (array-like): The input array to check for the middle number.

    Returns:
    -------
        array-like: A boolean mask indicating the middle number based on
                    certain conditions.
    """
    indices = np.arange(len(array) - 1)
    tmp = indices + np.argmax(array[indices: indices + 2], axis=1)
    mask = np.zeros(len(array), dtype=np.bool_)
    mask[tmp] = 1

    return mask


def scale_range(input, min: float, max: float):
    """
    Scales the input array to a specified range using min-max scaling.

    Args:
    ----
        input (array-like): The input array to be scaled.
        min (float): The minimum value of the output range.
        max (float): The maximum value of the output range.

    Returns:
    -------
        array-like: The scaled array within the specified range.
    """

    # min-max scaling
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def gc_skew(seq: str, window: int = 2048):
    """
    Calculates the GC skew along a DNA sequence with a specified window size.

    Args:
    ----
        seq (str): The DNA sequence for GC skew calculation.
        window (int, optional): The window size for calculating GC skew.
                                Defaults to 2048.

    Returns:
    -------
        dict: A dictionary containing the GC skew, position, and cumulative
              GC skew.
    """

    gc_skew = []
    lengths = []

    # lagging strand with negative GC skew.
    for i in range(0, len(seq) - window + 1, window):
        g = seq.count("G", i, i + window)
        c = seq.count("C", i, i + window)
        gc_skew.append(safe_divide((g - c), (g + c)))
        lengths.append(i)
    gc_skew = scale_range(
        np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same"),
        min=-1,
        max=1
    )
    cumsum = scale_range(np.cumsum(gc_skew), min=-1, max=1)
    return {"gc_skew": gc_skew,
            "position": np.array(lengths),
            "cum_gc": cumsum}


def logits_to_df(
    args,
    logits,
    headers,
    lengths,
    config,
    gc_skews,
    gcs,
    cutoff_length=500_000
):
    """
    Convert logits to a dict of dataframe for prophage region identification.
    Output of this function serves as the input for change point based
    segmentation.

    Args:
    ----
        logits: list of numpy arrys
        headers: numpy array of sequence identifiers of contigs

    Returns:
    -------
        tmp : Dict of [pandas dataframes, str:host, int:lengths]
    """
    lab = {int(k): v for k, v in config[args.model]["all_labels"].items()}
    tmp = {}
    for key, value, length, gc_skew, gc in zip(headers,
                                               logits,
                                               lengths,
                                               gc_skews,
                                               gcs):
        if length >= cutoff_length:
            try:
                value = np.exp(value) / np.sum(np.exp(value),
                                               axis=1).reshape(-1, 1)
                # bac, phage, euk, arch
                max_class = np.argmax(np.mean(value, axis=0))
                host = lab[max_class]
                t = pd.DataFrame(
                    value,
                    columns=list(config[args.model]["all_labels"].values())
                )
                t = t.assign(length=[i * args.fsize for i in range(len(t))])

                for k, v in lab.items():
                    t[v] = np.convolve(value[:, k], np.ones(4), mode="same")
                t["gc"] = gc
                t["gc_skew"] = scale_range(
                    np.convolve(np.array(gc_skew),
                                np.ones(10) / 10, mode="same"),
                    min=-1,
                    max=1,
                )

                tmp[f"{key}"] = [t, host, length]
            except Exception as e:
                logger.error(e)
                logger.debug(traceback.format_exc())

    return tmp


def plot_scores(logits_df, args, config, outdir, phage_cordinates):
    """
    Creates a circos plot of the host genome including putative prophages
    identified by Jaeger.

    Args:
    ----
        logits_df: DataFrame containing the logits.
        args: Dictionary of arguments.
        config: Dictionary containing configuration settings.
        outdir: Output directory for saving the plot.
        phage_cordinates: Dictionary of phage coordinates.

    Returns:
    -------
        None
    """
    # quantile cut-off 0.975 (or 0.025 of the right tail)
    lab = {int(k): v for k, v in
           config[args.model]["all_labels"].items()}
    # legend_lines = []

    # Plot outer track with xticks
    major_ticks_interval = 500_000
    minor_ticks_interval = 100_000

    for contig_id in logits_df.keys():

        tmp, host, length = logits_df[contig_id]
        circos = Circos(sectors={contig_id: length})
        sector = circos.get_sector(contig_id)

        outer_track = sector.add_track((98, 100))
        outer_track.axis(fc="lightgrey")
        outer_track.xticks_by_interval(
            major_ticks_interval,
            label_formatter=lambda v: f"{v/ 10 ** 6:.1f} Mb",
            show_endlabel=False,
            label_size=11,
        )

        outer_track.xticks_by_interval(
            minor_ticks_interval,
            tick_length=1,
            show_label=False,
            label_size=11
        )
        colors = ["gray", "green", "red", "teal", "brown"]
        patches = []

        for j, v in enumerate(lab.values()):
            # Plot Forward phage, bacterial, archaeal and eukaryotic scores
            if v == "phage":
                phage_track = sector.add_track((88, 97), r_pad_ratio=0.1)
                phage_track.fill_between(
                    tmp["length"],
                    tmp[v].to_numpy(),
                    vmin=0,
                    vmax=4,
                    color="orange",
                    alpha=1,
                )

                for cords in phage_cordinates[contig_id][0]:
                    pcs = np.arange(cords[0], cords[-1]) * args.fsize
                    phage_track.fill_between(
                        pcs,
                        np.ones_like(pcs) * 4,
                        vmin=0,
                        vmax=4,
                        color="magenta",
                        alpha=0.3,
                        lw=1,
                    )
            else:
                aux_track = sector.add_track((78, 87), r_pad_ratio=0.1)
                aux_track.fill_between(
                    tmp["length"],
                    tmp[v].to_numpy(),
                    vmin=0,
                    vmax=4,
                    color=colors[j],
                    alpha=0.7,
                )
                patches.append(Patch(color=colors[j], label=v))

        # Plot G+C
        gc_content_track = sector.add_track((55, 70))
        tmp["gc"] = tmp["gc"] - tmp["gc"].mean()
        positive_gc_contents = np.where(tmp["gc"] > 0, tmp["gc"], 0)
        negative_gc_contents = np.where(tmp["gc"] < 0, tmp["gc"], 0)
        abs_max_gc_content = np.max(np.abs(tmp["gc"]))

        vmin, vmax = -abs_max_gc_content, abs_max_gc_content
        gc_content_track.fill_between(
            tmp["length"],
            positive_gc_contents,
            0,
            vmin=vmin,
            vmax=vmax,
            color="blue",
            alpha=0.5,
        )
        gc_content_track.fill_between(
            tmp["length"],
            negative_gc_contents,
            0,
            vmin=vmin,
            vmax=vmax,
            color="black"
        )

        # Plot GC skew
        gc_skew_track = sector.add_track((45, 55))
        positive_gc_skews = np.where(tmp["gc_skew"] > 0, tmp["gc_skew"], 0)
        negative_gc_skews = np.where(tmp["gc_skew"] < 0, tmp["gc_skew"], 0)
        abs_max_gc_skew = np.max(np.abs(tmp["gc_skew"]))
        vmin, vmax = -abs_max_gc_skew, abs_max_gc_skew
        gc_skew_track.fill_between(
            tmp["length"],
            positive_gc_skews,
            0,
            vmin=vmin,
            vmax=vmax,
            color="olive"
        )
        gc_skew_track.fill_between(
            tmp["length"],
            negative_gc_skews,
            0,
            vmin=vmin,
            vmax=vmax,
            color="purple"
        )

        _ = circos.plotfig()
        plt.title(
            f"{contig_id.replace('__',',')}",
            fontdict={"size": 14, "weight": "bold"}
        )
        # Add legend
        handles = (
            [
                Patch(color="orange", label="phage"),
                Patch(color="magenta", alpha=0.3, label="putative prophage"),
            ]
            + patches
            + [
                Line2D(
                    [],
                    [],
                    color="blue",
                    label="$ > \overline{G+C}$",
                    marker="^",
                    ms=6,
                    ls="None",
                    alpha=0.5,
                ),
                Line2D(
                    [],
                    [],
                    color="black",
                    label="$ < \overline{G+C}$",
                    marker="v",
                    ms=6,
                    ls="None",
                ),
                Line2D(
                    [],
                    [],
                    color="olive",
                    label="Positive GC Skew",
                    marker="^",
                    ms=6,
                    ls="None",
                ),
                Line2D(
                    [],
                    [],
                    color="purple",
                    label="Negative GC Skew",
                    marker="v",
                    ms=6,
                    ls="None",
                ),
            ]
        )
        _ = circos.ax.legend(
            handles=handles,
            bbox_to_anchor=(0.51, 0.50),
            loc="center",
            fontsize=11
        )

        plt.savefig(
            os.path.join(
                outdir,
                f'{os.path.splitext(os.path.basename(args.input))[0]}_jaeger_{contig_id.split(" ")[0]}.pdf',
            ),
            bbox_inches="tight",
            dpi=300,
        )
        logger.info(
            (
                "prophage plot saved at "
                + os.path.join(
                    outdir,
                    f'{os.path.splitext(os.path.basename(args.input))[0]}_jaeger_{contig_id.split(" ")[0]}.pdf',
                )
            )
        )
        plt.close()


def segment(
    logits_df,
    outdir,
    cutoff_length=500_000,
    sensitivity=1.5,
    identifier="phage"
):
    """
    Segments the logit arrays based on change point detection and a
    sensitivity threshold.

    Args:
    ----
        logits_df (dict): Dictionary containing data for segmentation.
        outdir (str): Output directory for saving segmentation results.
        cutoff_length (int, optional): Length threshold for segmenting
                                       data. Defaults to 500,000.
        sensitivity (float, optional): Sensitivity threshold for segmentation.
                                       Defaults to 1.5.

    Returns:
    -------
        dict: A dictionary containing segmented data coordinates and scores.
    """
    phage_cordinates = {}
    for key, (tmp, host, length) in logits_df.items():
        if length <= cutoff_length:
            continue

        try:
            algo = rpt.KernelCPD(kernel="linear", min_size=3, jump=1).fit(
                tmp[identifier].to_numpy()
            )
            if bkpts := [
                algo.predict(pen=i)
                for i in range(1, 10)
                if len(algo.predict(pen=i)) > 1
            ]:
                bkpt_lens = np.array([len(b) for b in bkpts])
                kn = KneeLocator(
                    bkpt_lens,
                    list(range(len(bkpts))),
                    curve="convex",
                    direction="decreasing",
                )
                bkpt_index = (
                    [len(b) for b in bkpts].index(kn.knee)
                    if kn.knee
                    else np.searchsorted(bkpt_lens, 1)
                )
                if bkpt_index == len(bkpt_lens):
                    bkpt_index = None

                # all_high_indices = tmp[
                #     tmp[identifier] > np.quantile(tmp[identifier], q=0.975)
                # ].index.to_numpy()
                ranges = [
                    bkpts[bkpt_index][i: i + 2]
                    for i in range(len(bkpts[bkpt_index]) - 1)
                ]
                range_scores = np.array(
                    [tmp.loc[s:e][identifier].mean() for s, e in ranges]
                )
                range_mask = range_scores > sensitivity
                selected_ranges = merge_overlapping_ranges(
                    np.array(ranges)[range_mask])
                selected_ranges = np.array(selected_ranges)
                # nw_bkpts = np.append(
                #     selected_ranges.flatten(),
                # tmp[[identifier]].to_numpy().shape[0]
                # )

                phage_cordinates[key] = [selected_ranges,
                                         range_scores[range_mask]]
            else:
                phage_cordinates[key] = [[], []]
        except Exception:
            phage_cordinates[key] = [[], []]
            logger.debug(traceback.format_exc())

    return phage_cordinates


def calculate_gc_content(sequence):
    """
    Calculates the GC content of a given DNA sequence.

    Args:
    ----
        sequence (str): The DNA sequence for which GC content is calculated.

    Returns:
    -------
        float: The GC content of the DNA sequence.
    """

    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def calculate_percentage_of_n(sequence):
    """
    Calculates the percentage of 'N' bases in a given DNA sequence.

    Args:
    ----
        sequence (str): The DNA sequence

    Returns:
    -------
        float: proportion of Ns
    """

    return sequence.count("N") / len(sequence)


def get_prophage_alignment_summary(
    result_object,
    seq_len, record,
    cordinates,
    phage_score,
    type_="DTR"
):
    """
    Generates a summary of the prophage alignment results.

    Args:
    ----
        result_object: The alignment result object.
        seq_len: The length of the DNA sequence.
        record: The DNA sequence record.
        cordinates: The start and end coordinates for alignment.
        phage_score: The score of the prophage.
        type_ (str, optional): The type of prophage repeat. Defaults to "DTR".

    Returns:
    -------
        dict or str: A dictionary containing the prophage
    """

    if result_object is None:
        s_alig_start = cordinates["start"][0]
        e_alig_end = cordinates["end"][0]
        sequence = record[1][s_alig_start:e_alig_end]
        gc_ = calculate_gc_content(sequence)

        return {
            "contig_id": record[0],
            "alignment_length": None,
            "identities": None,
            "identity": None,
            "score": None,
            "type": None,
            "fgaps": None,
            "rgaps": None,
            "sstart": s_alig_start,
            "send": None,
            "estart": None,
            "eend": e_alig_end,
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": None,
            "gc%": gc_,
            "reject": None,
            "attL": None,
            "attR": None,
        }
    elif result_object.saturated:
        return "saturated"

    else:
        alig_len = len(result_object.traceback.query)
        f_gaps = result_object.traceback.query.count("-")
        rc_gaps = result_object.traceback.ref.count("-")
        iden = result_object.traceback.comp.count("|")

        ltr_cutoff = 250

        if type_ == "ITR":
            s_alig_end = cordinates["start"][0] + result_object.end_query + 1
            s_alig_start = s_alig_end - alig_len
            e_alig_start = cordinates["end"][1] - result_object.end_ref - 1
            e_alig_end = e_alig_start + alig_len
        elif type_ == "DTR":
            s_alig_end = cordinates["start"][0] + result_object.end_query
            s_alig_start = s_alig_end - alig_len + 1
            e_alig_end = cordinates["end"][0] + result_object.end_ref
            e_alig_start = e_alig_end - alig_len + 1

            if (s_alig_end - s_alig_start) >= ltr_cutoff:
                type_ = f"LTR_{type_}"

        sequence = record[1][s_alig_start:e_alig_end]
        percentage_of_N = calculate_percentage_of_n(sequence)
        gc_ = calculate_gc_content(sequence)

        return {
            "contig_id": record[0],
            "alignment_length": alig_len,
            "identities": iden,
            "identity": round(iden / alig_len, 2),
            "score": result_object.score,
            "type": type_,
            "fgaps": f_gaps,
            "rgaps": rc_gaps,
            "sstart": s_alig_start,
            "send": s_alig_end,
            "estart": e_alig_start,
            "eend": e_alig_end,
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": percentage_of_N,
            "gc%": gc_,
            "reject": percentage_of_N > 0.20,
            "attL": result_object.traceback.query,
            "attR": result_object.traceback.ref,
        }


def prophage_report(args, filehandle, prophage_cordinates, outdir):
    """
    Searches for direct repeats at prophage boundaries and generates
    prophage summaries.

    Args:
    ----
        args: Arguments for the search process.
        filehandle: File handle for reading DNA sequences.
        prophage_cordinates: Coordinates of prophages for comparison.
        outdir: Output directory for saving prophage summaries.
    Returns
    -------
        None
    """

    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    summaries = []

    def append_summary(result, seq_len, record, start, end, j, type_):
        summaries.append(
            get_prophage_alignment_summary(
                result_object=result,
                seq_len=seq_len,
                record=record,
                cordinates={
                    "start": [start, start + off_set],
                    "end": [end - off_set, end + scan_length],
                },
                phage_score=j,
                type_=type_,
            )
        )

    try:
        for record in pyfastx.Fasta(filehandle, build_index=False):
            seq_len = len(record[1])
            header = record[0].replace(",", "__")
            logger.debug(f'generating prophage report for {header}')
            if seq_len > 500_000:
                cords, scores = prophage_cordinates.get(f"{header}", [[], []])
                if len(cords) > 0 and len(scores) > 0:
                    for (start, end), j in zip(cords, scores):
                        start, end = start * args.fsize, end * args.fsize
                        scan_length = min(max(int(seq_len * 0.04), 400), 4000)
                        off_set = (
                            2000 if (end - start) // 2 >= 14000
                            else (end - start) // 4
                        )

                        logger.info(
                            f"searching for direct repeats {start - scan_length}:{start + off_set} | {end - off_set}:{end + scan_length}"
                        )

                        result_dtr = parasail.sw_trace_scan_16(
                            str(record[1][start - scan_length:
                                          start + off_set]),
                            str(record[1][end - off_set:
                                          end + scan_length]),
                            100,
                            5,
                            user_matrix,
                        )

                        result_itr = parasail.sw_trace_scan_16(
                            str(record[1][start - scan_length:
                                          start + off_set]),
                            reverse_complement(
                                str(record[1][end - off_set:
                                              end + scan_length])
                            ),
                            100,
                            5,
                            user_matrix,
                        )

                        if (
                            len(result_itr.traceback.query) > 12
                            or len(result_dtr.traceback.query) > 12
                        ):
                            if result_itr.score > result_dtr.score:
                                append_summary(
                                    result_itr,
                                    seq_len,
                                    record,
                                    start - scan_length,
                                    end + scan_length,
                                    j,
                                    "ITR",
                                )
                            else:
                                append_summary(
                                    result_dtr,
                                    seq_len,
                                    record,
                                    start - scan_length,
                                    end + scan_length,
                                    j,
                                    "DTR",
                                )
                        else:
                            summaries.append(
                                get_prophage_alignment_summary(
                                    result_object=None,
                                    seq_len=seq_len,
                                    record=record,
                                    cordinates={
                                        "start": [start, None],
                                        "end": [end, None],
                                    },
                                    phage_score=j,
                                    type_=None,
                                )
                            )

        if summaries:
            df = pd.DataFrame(summaries)
            df["contig_id"] = df["contig_id"].apply(lambda x:
                                                    x.replace("__", ","))
            df.to_csv(
                os.path.join(outdir, "prophages_jaeger.tsv"),
                sep="\t",
                index=False
            )
            logger.info(f"prophage cordinates saved at {os.path.join(outdir, 'prophages_jaeger.tsv')}")

    except Exception as e:
        logger.error(f"an error {e} occured during prophage report generation")
        logger.debug(traceback.format_exc())


def get_alignment_summary(result_object,
                          seq_len,
                          record_id,
                          input_length,
                          type_="DTR"):
    """
    Generates a summary of the alignment results from parasail.

    Args:
    ----
        result_object: The alignment result object.
        seq_len: The length of the DNA sequence.
        record_id: The identifier of the DNA sequence.
        input_length: The input length for alignment.
        type_ (str, optional): The type of terminal repeat. Defaults to "DTR".

    Returns:
    -------
        dict: A dictionary containing the alignment summary details.
    """

    if result_object.saturated:
        return "saturated"

    alig_len = len(result_object.traceback.query)
    f_gaps = result_object.traceback.query.count("-")
    rc_gaps = result_object.traceback.ref.count("-")
    iden = result_object.traceback.comp.count("|")

    ltr_cutoff = 250

    # Common calculations
    s_alig_start = (result_object.end_query - alig_len + f_gaps) + 1
    s_alig_end = result_object.end_query + 1

    if type_ == "ITR":
        e_alig_start = (seq_len - input_length) + max(
            input_length - result_object.end_ref, 0
        )
        e_alig_end = e_alig_start + (alig_len - rc_gaps)
        rear = reverse_complement(str(result_object.traceback.ref))
    elif type_ == "DTR":
        e_alig_start = (seq_len - input_length) + max(
            result_object.end_ref - alig_len, 0
        )
        e_alig_end = (seq_len - input_length) + result_object.end_ref
        if (s_alig_end - s_alig_start) >= ltr_cutoff:
            type_ = f"LTR_{type_}"
        rear = result_object.traceback.ref

    return {
        "contig_id": record_id,
        "repeat_length": alig_len,
        "identities": iden,
        "identity": safe_divide(iden, alig_len),
        "score": result_object.score,
        "terminal_repeats": type_,
        "fgaps": f_gaps,
        "rgaps": rc_gaps,
        "sstart": s_alig_start,
        "send": s_alig_end,
        "estart": e_alig_start,
        "eend": e_alig_end,
        "seq_len": seq_len,
        "front": result_object.traceback.query,
        "rear": rear,
    }


def scan_for_terminal_repeats(args, file_path, num):
    """
    Scans for terminal repeats in DNA sequences using Smith-Waterman alignment.

    Args:
    ----
        args: Arguments for the scanning process.
        infile: Input file containing DNA sequences.
        num: Number of sequences to scan.

    Returns:
    -------
        pandas.DataFrame: A DataFrame containing the summary of terminal
        repeat scans.
    """

    # ideally DRs should be found in the intergenic region
    logger.info("scaning for terminal repeats")
    user_matrix = parasail.matrix_create("ACGT", 2, -100)
    summaries = []

    def helper(record):

        seq_len = len(record[1])
        headder = record[0].replace(",", "__")
        logger.debug(f"{headder}, {seq_len}")
        scan_length = min(max(int(seq_len * 0.04), 400), 4000)

        result_itr = parasail.sw_trace_scan_16(
            str(record[1][:scan_length]),
            reverse_complement(record[1][-scan_length:]),
            100,
            5,
            user_matrix,
        )

        result_dtr = parasail.sw_trace_scan_16(
            str(record[1][:scan_length]),
            str(record[1][-scan_length:]),
            100,
            5,
            user_matrix,
        )
        if len(result_itr.traceback.query) > 12 or\
           len(result_dtr.traceback.query) > 12:
            if result_itr.score > result_dtr.score:
                return get_alignment_summary(
                    result_object=result_itr,
                    seq_len=seq_len,
                    record_id=headder,
                    input_length=scan_length,
                    type_="ITR",
                )
            else:
                return get_alignment_summary(
                    result_object=result_dtr,
                    seq_len=seq_len,
                    record_id=headder,
                    input_length=scan_length,
                    type_="DTR",
                )
        else:
            return {
                "contig_id": headder,
                "repeat_length": None,
                "identities": None,
                "identity": None,
                "score": None,
                "terminal_repeats": None,
                "fgaps": None,
                "rgaps": None,
                "sstart": None,
                "send": None,
                "estart": None,
                "eend": None,
                "seq_len": seq_len,
                "front": None,
                "rear": None,
            }

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(helper, record)
                for record in pyfastx.Fasta(file_path, build_index=False)
                if len(record[1]) >= args.fsize
                ]

            # Retrieve and print the results
            # pbar = tqdm(total=num,
            #             ascii=" >=",
            #             bar_format="{l_bar}{bar:40}{r_bar}",
            #             dynamic_ncols=True,
            #             unit="seq",
            #             colour="green",
            #             position=0,
            #             leave=True,
            #             redirect_stdout=True
            #             )
            with progressbar.ProgressBar(max_value=num) as pbar:
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    summaries.append(result)
                    pbar.update(i)

    except RuntimeError as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    return pd.DataFrame(summaries)
