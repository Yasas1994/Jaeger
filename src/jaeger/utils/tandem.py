import subprocess
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shutil
import logging
# Paths to executables
TRF_EXECUTABLE = "trf"       # Adjust if TRF is not in PATH
PYFASTX_EXECUTABLE = "pyfastx"  # Adjust if pyfastx is not in PATH

# TRF parameters
MATCH, MISMATCH, DELTA, PM, PI, MINSCORE, MAXPERIOD = 2, 7, 7, 80, 10, 50, 500

# trf jaeger_predictions_trf/1.5M/chunks/GCF_022682495.2_HLdesRot8A.1_genomic.01.fna 2 5 7 80 10 50 2000 -m -h
logger = logging.getLogger("Jaeger")

def split_fasta_with_pyfastx(input_fasta: str, output_dir: str, chunks: int = None, counts: int = None):
    """
    Use pyfastx split to split a fasta file into chunks and remove empty splits.

    Parameters
    ----------
    input_fasta : str
        Path to input FASTA
    output_dir : str
        Output directory for chunk files
    chunks : int, optional
        Split into N equal chunks
    counts : int, optional
        Split by number of sequences per chunk

    Returns
    -------
    list of str
        Paths to non-empty chunk FASTA files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pyfastx split
    if chunks:
        cmd = [PYFASTX_EXECUTABLE, "split", "-n", str(chunks), "-o", str(output_dir), input_fasta]
    elif counts:
        cmd = [PYFASTX_EXECUTABLE, "split", "-c", str(counts), "-o", str(output_dir), input_fasta]
    else:
        raise ValueError("You must specify either chunks (-n) or counts (-c)")

    logger.info("Splitting FASTA with pyfastx...")
    subprocess.run(cmd, check=True)

    # Collect chunk files
    all_chunks = list(output_dir.glob(f"*{Path(input_fasta).suffix}"))

    non_empty_chunks = []
    for f in sorted(all_chunks):
        if f.stat().st_size == 0:
            # Remove empty file
            print(f"[!] Removing empty split: {f}")
            f.unlink()
        else:
            non_empty_chunks.append(str(f))

    logger.info(f"Split into {len(non_empty_chunks)} non-empty chunk files.")
    # print(f"Split into {len(non_empty_chunks)} non-empty chunk files.")
    return non_empty_chunks


def run_trf_and_collect_mask(fasta_path: str, out_dir: str):
    """
    Run TRF on a single FASTA chunk and collect the masked file.
    """
    fasta_path = Path(fasta_path).resolve()
    fasta_name = fasta_path.name
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        TRF_EXECUTABLE, str(fasta_path),
        str(MATCH), str(MISMATCH), str(DELTA),
        str(PM), str(PI), str(MINSCORE), str(MAXPERIOD),
        "-h", "-m"
    ]
    subprocess.run(cmd, check=True, cwd=out_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    param_suffix = f".{MATCH}.{MISMATCH}.{DELTA}.{PM}.{PI}.{MINSCORE}.{MAXPERIOD}.mask"
    mask_file = out_dir/ f"{fasta_name}{param_suffix}"

    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")


    # Clean up TRF auxiliary files
    for ext in [".dat", ".repeat"]:
        extra = fasta_name + f".{MATCH}.{MISMATCH}.{DELTA}.{PM}.{PI}.{MINSCORE}.{MAXPERIOD}{ext}"
        if os.path.exists(extra):
            os.remove(extra)

    return mask_file


def run_batch(fasta_files, out_dir="masked_chunks", n_threads=None):
    """
    Run TRF on a list of fasta files in parallel.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_threads is None:
        n_threads = max(1, cpu_count() - 1)

    args = [(f, out_dir) for f in fasta_files]
    with Pool(processes=n_threads) as pool:
        results = pool.starmap(run_trf_and_collect_mask, args)

     
    logger.debug(f"All masked chunk files saved to: {out_dir}")
    return results


def merge_masked_files(masked_files, output_file):
    """
    Merge all masked fasta files into a single fasta file.
    """
    with open(output_file, "w") as outfile:
        for mf in masked_files:
            with open(mf) as infile:
                shutil.copyfileobj(infile, outfile)
    logger.debug(f"Merged masked FASTAs into: {output_file}")
    #print(f"Merged masked FASTAs into: {output_file}")

if __name__ == "__main__":
    # Example usage
    from sys import argv
    from datetime import datetime
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d%H%M%S")
    input_fasta = argv[1]
    chunks_dir = f"tmp_chunks_{timestamp_str}"
    masked_chunks_dir =  f"masked_chunks_{timestamp_str}"
    final_masked = argv[2]

    # 1. Split the input FASTA
    chunk_files = split_fasta_with_pyfastx(input_fasta, chunks_dir, chunks=16)
    # Or: chunk_files = split_fasta_with_pyfastx(input_fasta, chunks_dir, chunks=10)

    # 2. Run TRF on chunks in parallel
    masked_files = run_batch(chunk_files, out_dir=masked_chunks_dir, n_threads=16)

    # 3. Merge all masked files into one
    merge_masked_files(masked_files, final_masked)

    # 4. Cleanup temporary directories
    shutil.rmtree(chunks_dir)
    shutil.rmtree(masked_chunks_dir)
    logger.debug("Cleaned up temporary files.")
    logger.debug("Done. Final masked FASTA: {final_masked}")



