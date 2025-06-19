import pyfastx
import pydustmasker
from rich.progress import Progress
from jaeger.utils.misc import signal_l, safe_divide


def fragment_generator(
    file_path, fragsize=None, stride=None, num=None, no_progress=True, dustmask=True
):
    """
    Generates fragments of DNA sequences from a FASTA file.
    optionally masks low complexity regions (recomemded to use when
    analysing eukaryotic host associated metagenomes)

    Args:
    ----
        filehandle: File handle for reading DNA sequences.
        fragsize (int, optional): Size of the DNA sequence fragments. Defaults
        to None.
        stride (int, optional): Stride for fragment generation. Defaults to
        None.
        num (int, optional): Total number of sequences to process. Defaults
        to None.
        no_progress (bool, optional): Flag to disable progress bar. Defaults to
        False.
        dustmask (bool, optional): Flag to enable dust masking. Default to True
    Returns:
    -------
        generator: A generator that yields DNA sequence fragments with
        associated information.
    """

    def c():
        # accepts a reference to a file handle
        fa = pyfastx.Fasta(file_path, build_index=False)
        with Progress(transient=True, disable=no_progress) as progress:
            task = progress.add_task("[cyan]Reading fasta...", total=num)
            for j, record in enumerate(fa):
                progress.update(task, advance=1)
                seqlen = len(
                    record[1]
                )  # move size filtering to a separate preprocessing step
                sequence = record[1].strip().upper()
                if dustmask:
                    sequence = pydustmasker.DustMasker(sequence).mask()
                header = record[0].strip().replace(",", "__")
                # logger.debug(sequence)
                # sequence = str(record[1]).upper()
                # filters the sequence based on size
                if seqlen >= fragsize:
                    # if no fragsize, return the entire sequence
                    if fragsize is None:
                        # sequence and sequence headder
                        yield f"{sequence},{header}"
                    elif fragsize is not None:
                        for i, (b, index) in enumerate(
                            signal_l(
                                range(
                                    0,
                                    seqlen - (fragsize - 1),
                                    fragsize if stride is None else stride,
                                )
                            )
                        ):
                            g = sequence[index : index + fragsize].count("G")
                            c = sequence[index : index + fragsize].count("C")
                            a = sequence[index : index + fragsize].count("A")
                            t = sequence[index : index + fragsize].count("T")
                            gc_skew = safe_divide((g - c), (g + c))
                            # sequnce_fragment, contig_id, index, contig_end, i,
                            # g, c, gc_skew
                            yield f"{sequence[index : index + fragsize]},{header},{index},{b},{i},{seqlen},{g},{c},{a},{t},{gc_skew : .3f}"

    return c


def fragment_generator_lib(filename, fragsize=None, stride=None, num=None):
    """
    Generates fragments of DNA sequences from various input types.

    Args:
    ----
        filehandle: Input source for DNA sequences, can be a file handle,
                    string, list, Seq object, generator, or file object.
        fragsize (int, optional): Size of the DNA sequence fragments.
                                  Defaults to None.
        stride (int, optional): Stride for fragment generation.
                                Defaults to None.
        num (int, optional): Total number of sequences to process.
                             Defaults to None.

    Returns:
    -------
        generator: A generator that yields DNA sequence fragments with
                   associated information.

    Raises:
    ------
        ValueError: If the input type is not supported.
    """

    head = False
    if isinstance(filename, str):
        tmpfn = pyfastx.Fasta(filename, build_index=False)
        head = True
    else:
        raise ValueError("Not a supported input type")

    def c():
        # accepts a reference to a file handle
        for n, record in enumerate(tmpfn):
            if head:
                seqlen = len(
                    record[1]
                )  # move size filtering to a separate preprocessing step
                seq = record[1]
                headder = record[0].replace(",", "__")

            else:
                seqlen = len(record)
                seq = record
                headder = f"seq_{n}"
            # filters the sequence based on size
            if seqlen >= fragsize:
                # if no fragsize, return the entire sequence
                if fragsize is None:
                    yield f"{str(seq)},{str(headder)}"
                elif fragsize is not None:
                    for i, (b, index) in enumerate(
                        signal_l(
                            range(
                                0,
                                seqlen - (fragsize - 1),
                                fragsize if stride is None else stride,
                            )
                        )
                    ):
                        yield f"{str(seq)[index:index + fragsize]},{str(headder)},{str(index)},{str(b)},{str(i)},{str(seqlen)}"

    return c
