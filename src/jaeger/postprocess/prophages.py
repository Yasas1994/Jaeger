"""

Copyright (c) 2024 Yasas Wijesekara

"""

import os
import logging
import traceback
import pyfastx
import parasail
import numpy as np
import ruptures as rpt
import pandas as pd
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kneed import KneeLocator
from pycirclize import Circos
from jaeger.postprocess.helpers import (
    calculate_gc_content,
    calculate_percentage_of_n,
    merge_overlapping_ranges,
    scale_range,
)
from jaeger.seqops.transform import reverse_complement

logger = logging.getLogger("jaeger")


def logits_to_df(config: Any, cmdline_kwargs: dict, **kwargs) -> dict:
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
        tmp : dict of [pandas dataframes, str:host, int:lengths]
    """
    lab = {int(k): v for k, v in config["all_labels"].items()}
    tmp = {}
    for key, value, length, gc_skew, gc in zip(
        kwargs.get("headers"),
        kwargs.get("predictions"),
        kwargs.get("lengths"),
        kwargs.get("gc_skews"),
        kwargs.get("gcs"),
    ):
        if length >= cmdline_kwargs.get("lc"):
            # try:
            value = np.exp(value) / np.sum(np.exp(value), axis=1).reshape(-1, 1)
            # bac, phage, euk, arch
            max_class = np.argmax(np.mean(value, axis=0))
            host = lab[max_class]
            t = pd.DataFrame(
                value,
                columns=list(config["all_labels"].values()),
            )
            t = t.assign(
                length=[i * cmdline_kwargs.get("fsize") for i in range(len(t))]
            )

            for k, v in lab.items():
                conv = np.convolve(value[:, k], np.ones(4), mode="same")
                # trim/pad to match the number of windows
                if len(conv) > len(t):
                    conv = conv[: len(t)]
                elif len(conv) < len(t):
                    conv = np.pad(conv, (0, len(t) - len(conv)), mode="edge")
                t[v] = conv
            t["gc"] = gc[: len(t)] if len(gc) > len(t) else gc
            gc_skew_conv = np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same")
            if len(gc_skew_conv) > len(t):
                gc_skew_conv = gc_skew_conv[: len(t)]
            elif len(gc_skew_conv) < len(t):
                gc_skew_conv = np.pad(
                    gc_skew_conv, (0, len(t) - len(gc_skew_conv)), mode="edge"
                )
            t["gc_skew"] = scale_range(gc_skew_conv, min=-1, max=1)

            tmp[f"{key}"] = [t, host, length]
        # except Exception as e:
        #     logger.error(e)
        #     logger.debug(traceback.format_exc())

    return tmp


def logits_to_df_v2(
    class_map: dict,
    cmdline_kwargs: dict,
    headers: np.ndarray,
    predictions: list[np.ndarray],
    lengths: np.ndarray,
    gc_skews: list[np.ndarray],
    gcs: list[np.ndarray],
) -> dict:
    """Convert logits to a dict of dataframes for prophage region identification.

    Compatible with the new SavedModel / TFLite / ONNX pipeline.
    Uses ``class_map`` (with ``class`` and ``index`` keys) instead of the
    legacy ``config["all_labels"]`` dict.

    Returns:
        dict mapping contig_id -> [DataFrame, host_label, length]
    """
    indices = class_map.get("index", [])
    classes = class_map.get("class", [])
    lab = {int(i): c for i, c in zip(indices, classes)}

    tmp = {}
    for key, value, length, gc_skew, gc in zip(
        headers, predictions, lengths, gc_skews, gcs
    ):
        if length >= cmdline_kwargs.get("lc", 500_000):
            value = np.exp(value) / np.sum(np.exp(value), axis=1).reshape(-1, 1)
            max_class = np.argmax(np.mean(value, axis=0))
            host = lab.get(max_class, "unknown")
            t = pd.DataFrame(value, columns=list(lab.values()))
            t = t.assign(
                length=[i * cmdline_kwargs.get("fsize", 2000) for i in range(len(t))]
            )
            for k, v in lab.items():
                conv = np.convolve(value[:, k], np.ones(4), mode="same")
                if len(conv) > len(t):
                    conv = conv[: len(t)]
                elif len(conv) < len(t):
                    conv = np.pad(conv, (0, len(t) - len(conv)), mode="edge")
                t[v] = conv
            t["gc"] = gc[: len(t)] if len(gc) > len(t) else gc
            gc_skew_conv = np.convolve(np.array(gc_skew), np.ones(10) / 10, mode="same")
            if len(gc_skew_conv) > len(t):
                gc_skew_conv = gc_skew_conv[: len(t)]
            elif len(gc_skew_conv) < len(t):
                gc_skew_conv = np.pad(
                    gc_skew_conv, (0, len(t) - len(gc_skew_conv)), mode="edge"
                )
            t["gc_skew"] = scale_range(gc_skew_conv, min=-1, max=1)
            tmp[f"{key}"] = [t, host, length]
    return tmp


def plot_scores(
    logits_df: pd.DataFrame,
    config: Any,
    model: str,
    fsize: int,
    infile_base: str,
    outdir: Path,
    phage_cordinates: dict,
) -> None:
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
    lab = {int(k): v for k, v in config["all_labels"].items()}
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
            label_formatter=lambda v: f"{v / 1e6:.1f} Mb",
            show_endlabel=False,
            label_size=11,
        )

        outer_track.xticks_by_interval(
            minor_ticks_interval, tick_length=1, show_label=False, label_size=11
        )
        colors = ["gray", "green", "red", "teal", "brown", "purple", "cyan", "pink"]
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
                    pcs = np.arange(cords[0], cords[-1]) * fsize
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
                color = colors[j % len(colors)]
                aux_track = sector.add_track((78, 87), r_pad_ratio=0.1)
                aux_track.fill_between(
                    tmp["length"],
                    tmp[v].to_numpy(),
                    vmin=0,
                    vmax=4,
                    color=color,
                    alpha=0.7,
                )
                patches.append(Patch(color=color, label=v))

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
            tmp["length"], negative_gc_contents, 0, vmin=vmin, vmax=vmax, color="black"
        )

        # Plot GC skew
        gc_skew_track = sector.add_track((45, 55))
        positive_gc_skews = np.where(tmp["gc_skew"] > 0, tmp["gc_skew"], 0)
        negative_gc_skews = np.where(tmp["gc_skew"] < 0, tmp["gc_skew"], 0)
        abs_max_gc_skew = np.max(np.abs(tmp["gc_skew"]))
        vmin, vmax = -abs_max_gc_skew, abs_max_gc_skew
        gc_skew_track.fill_between(
            tmp["length"], positive_gc_skews, 0, vmin=vmin, vmax=vmax, color="olive"
        )
        gc_skew_track.fill_between(
            tmp["length"], negative_gc_skews, 0, vmin=vmin, vmax=vmax, color="purple"
        )

        _ = circos.plotfig()
        plt.title(
            f"{contig_id.replace('___', ',')}", fontdict={"size": 14, "weight": "bold"}
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
            handles=handles, bbox_to_anchor=(0.51, 0.50), loc="center", fontsize=11
        )

        plt.savefig(
            os.path.join(
                outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}.pdf",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        logger.info(
            (
                "prophage plot saved at "
                + os.path.join(
                    outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}.pdf",
                )
            )
        )
        plt.close()


def plot_scores_linear(
    logits_df: pd.DataFrame,
    config: Any,
    model: str,
    fsize: int,
    infile_base: str,
    outdir: Path,
    phage_cordinates: dict,
) -> None:
    """
    Creates a linear genome plot of the host genome including putative
    prophages identified by Jaeger.

    Args:
    ----
        logits_df: DataFrame containing the logits.
        config: Dictionary containing configuration settings.
        model: Model identifier string.
        fsize: Fragment size in bp.
        infile_base: Base name of the input file.
        outdir: Output directory for saving the plot.
        phage_cordinates: Dictionary of phage coordinates.

    Returns:
    -------
        None
    """
    lab = {int(k): v for k, v in config["all_labels"].items()}
    colors = ["gray", "green", "red", "teal", "brown", "purple", "pink", "olive"]

    for contig_id in logits_df.keys():
        tmp, host, length = logits_df[contig_id]
        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1.5, 1.5], "hspace": 0.05},
        )
        ax_phage, ax_aux, ax_gc, ax_skew = axes

        # --- Phage score track (top) ---
        ax_phage.fill_between(
            tmp["length"],
            tmp["phage"].to_numpy(),
            color="orange",
            alpha=0.8,
            label="phage score",
        )
        # Highlight prophage regions
        for cords in phage_cordinates.get(contig_id, [[], []])[0]:
            pcs = np.arange(cords[0], cords[-1]) * fsize
            ax_phage.fill_between(
                pcs,
                np.ones_like(pcs) * 4,
                color="magenta",
                alpha=0.25,
                label="putative prophage"
                if cords[0] == phage_cordinates[contig_id][0][0][0]
                else "",
            )
        ax_phage.set_ylim(0, 4)
        ax_phage.set_ylabel("Phage score", fontsize=10)
        ax_phage.set_title(
            f"{contig_id.replace('___', ',')}",
            fontdict={"size": 12, "weight": "bold"},
        )
        ax_phage.legend(loc="upper right", fontsize=8)
        ax_phage.grid(True, linestyle=":", alpha=0.4)

        # --- Auxiliary class scores ---
        patches = []
        for j, v in enumerate(lab.values()):
            if v == "phage":
                continue
            ax_aux.plot(
                tmp["length"],
                tmp[v].to_numpy(),
                color=colors[j % len(colors)],
                alpha=0.7,
                linewidth=1.2,
                label=v,
            )
            patches.append(Patch(color=colors[j % len(colors)], label=v))
        ax_aux.set_ylim(0, 4)
        ax_aux.set_ylabel("Class scores", fontsize=10)
        ax_aux.legend(loc="upper right", fontsize=8)
        ax_aux.grid(True, linestyle=":", alpha=0.4)

        # --- GC content track ---
        gc_centered = tmp["gc"] - tmp["gc"].mean()
        positive_gc = np.where(gc_centered > 0, gc_centered, 0)
        negative_gc = np.where(gc_centered < 0, gc_centered, 0)
        abs_max_gc = np.max(np.abs(gc_centered))
        vmin_gc, vmax_gc = -abs_max_gc, abs_max_gc

        ax_gc.fill_between(
            tmp["length"],
            positive_gc,
            0,
            color="blue",
            alpha=0.5,
            label=r"$> \overline{G+C}$",
        )
        ax_gc.fill_between(
            tmp["length"],
            negative_gc,
            0,
            color="black",
            alpha=0.7,
            label=r"$< \overline{G+C}$",
        )
        ax_gc.set_ylim(vmin_gc, vmax_gc)
        ax_gc.set_ylabel("G+C dev.", fontsize=10)
        ax_gc.legend(loc="upper right", fontsize=8)
        ax_gc.grid(True, linestyle=":", alpha=0.4)

        # --- GC skew track ---
        positive_skew = np.where(tmp["gc_skew"] > 0, tmp["gc_skew"], 0)
        negative_skew = np.where(tmp["gc_skew"] < 0, tmp["gc_skew"], 0)
        abs_max_skew = np.max(np.abs(tmp["gc_skew"]))
        vmin_skew, vmax_skew = -abs_max_skew, abs_max_skew

        ax_skew.fill_between(
            tmp["length"],
            positive_skew,
            0,
            color="olive",
            alpha=0.7,
            label="Positive GC skew",
        )
        ax_skew.fill_between(
            tmp["length"],
            negative_skew,
            0,
            color="purple",
            alpha=0.7,
            label="Negative GC skew",
        )
        ax_skew.set_ylim(vmin_skew, vmax_skew)
        ax_skew.set_ylabel("GC skew", fontsize=10)
        ax_skew.set_xlabel("Genome position (bp)", fontsize=10)
        ax_skew.legend(loc="upper right", fontsize=8)
        ax_skew.grid(True, linestyle=":", alpha=0.4)

        # Format x-axis
        ax_skew.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f} Mb")
        )

        out_path = outdir / f"{infile_base}_jaeger_{contig_id.split(' ')[0]}_linear.pdf"
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        logger.info(f"linear prophage plot saved at {out_path}")
        plt.close()


def segment(
    logits_df: pd.DataFrame,
    outdir: Path,
    cutoff_length: int = 500_000,
    sensitivity: float = 1.5,
    identifier: str = "phage",
) -> dict:
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
                    bkpts[bkpt_index][i : i + 2]
                    for i in range(len(bkpts[bkpt_index]) - 1)
                ]
                range_scores = np.array(
                    [tmp.loc[s:e][identifier].mean() for s, e in ranges]
                )
                range_mask = range_scores > sensitivity
                selected_ranges = merge_overlapping_ranges(np.array(ranges)[range_mask])
                selected_ranges = np.array(selected_ranges)
                # nw_bkpts = np.append(
                #     selected_ranges.flatten(),
                # tmp[[identifier]].to_numpy().shape[0]
                # )

                phage_cordinates[key] = [selected_ranges, range_scores[range_mask]]
            else:
                phage_cordinates[key] = [[], []]
        except Exception:
            phage_cordinates[key] = [[], []]
            logger.debug(traceback.format_exc())

    return phage_cordinates


def get_prophage_alignment_summary(
    result_object, seq_len, record, cordinates, phage_score, type_="DTR"
) -> dict:
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
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": None,
            "gc%": gc_,
            "reject": None,
            "sstart": s_alig_start,
            "send": None,
            "estart": None,
            "eend": e_alig_end,
            "att_alignment_length": None,
            "att_identities": None,
            "att_identity": None,
            "att_score": None,
            "att_type": None,
            "att_fgaps": None,
            "att_rgaps": None,
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
            "seq_len": seq_len,
            "region_len": e_alig_end - s_alig_start,
            "phage_score": phage_score,
            "n%": percentage_of_N,
            "gc%": gc_,
            "reject": percentage_of_N > 0.20,
            "sstart": s_alig_start,
            "send": s_alig_end,
            "estart": e_alig_start,
            "eend": e_alig_end,
            "att_alignment_length": alig_len,
            "att_identities": iden,
            "att_identity": round(iden / alig_len, 2),
            "att_score": result_object.score,
            "att_type": type_,
            "att_fgaps": f_gaps,
            "att_rgaps": rc_gaps,
            "attL": result_object.traceback.query,
            "attR": result_object.traceback.ref,
        }


def prophage_report(
    fsize: int, filehandle: Any, prophage_cordinates: dict, outdir: Path
):
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

    for record in pyfastx.Fasta(filehandle, build_index=False):
        seq_len = len(record[1])
        header = record[0].replace(",", "___")
        logger.debug(f"generating prophage report for {header}")
        if seq_len > 500_000:
            cords, scores = prophage_cordinates.get(f"{header}", [[], []])
            if len(cords) > 0 and len(scores) > 0:
                for (start, end), j in zip(cords, scores):
                    start, end = start * fsize, end * fsize
                    scan_length = min(max(int(seq_len * 0.04), 400), 4000)
                    off_set = (
                        2000 if (end - start) // 2 >= 14000 else (end - start) // 4
                    )

                    # logger.info(
                    #     f"searching for direct repeats {start - scan_length}:{start + off_set} | {end - off_set}:{end + scan_length}"
                    # )

                    result_dtr = parasail.sw_trace_scan_16(
                        str(record[1][start - scan_length : start + off_set]),
                        str(record[1][end - off_set : end + scan_length]),
                        100,
                        5,
                        user_matrix,
                    )

                    result_itr = parasail.sw_trace_scan_16(
                        str(record[1][start - scan_length : start + off_set]),
                        reverse_complement(
                            str(record[1][end - off_set : end + scan_length])
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
        df["contig_id"] = df["contig_id"].apply(lambda x: x.replace("___", ","))
        df.to_csv(
            outdir / "prophages_jaeger.tsv", sep="\t", index=False, float_format="%.3f"
        )
        logger.info(f"prophage cordinates saved at {outdir / 'prophages_jaeger.tsv'}")
