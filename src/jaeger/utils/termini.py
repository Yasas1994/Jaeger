import sys
import logging
import traceback
from typing import Union, Any, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyfastx
import parasail
import pandas as pd
from rich.progress import Progress
from jaeger.utils.misc import safe_divide
from jaeger.utils.seq import reverse_complement

logger = logging.getLogger("jaeger")


def get_alignment_summary(
    result_object: Any,
    seq_len: int,
    record_id: str,
    input_length: int,
    type_: str = "DTR",
) -> Dict:
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


def scan_for_terminal_repeats(
    file_path: Union[str, Path], num: int, workers: int, fsize: int
) -> pd.DataFrame:
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
        headder = record[0].replace(",", "___")
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
        if len(result_itr.traceback.query) > 12 or len(result_dtr.traceback.query) > 12:
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(helper, record)
                for record in pyfastx.Fasta(file_path, build_index=False)
                if len(record[1]) >= fsize
            ]

            with Progress(transient=True) as progress:
                task = progress.add_task("[cyan]processing...", total=num)
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    summaries.append(result)
                    progress.update(task, advance=1)

    except RuntimeError as e:
        logger.error(e)
        logger.debug(traceback.format_exc())
        sys.exit(1)

    return pd.DataFrame(summaries)
