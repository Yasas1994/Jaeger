from pathlib import Path

from jaeger.seqops.io import _window_indices, fragment_generator


def test_window_indices_default_discards_tail():
    # 3400 bp contig, 2000 bp window, 2000 bp stride -> only one window.
    indices = _window_indices(
        seqlen=3400,
        fragsize=2000,
        stride=2000,
        dynamic_stride=False,
        dynamic_stride_threshold=2.0,
    )
    assert indices == [0]


def test_window_indices_dynamic_short_contig():
    # 3400 bp contig is below 2 * 2000 threshold, so two windows covering the end.
    indices = _window_indices(
        seqlen=3400,
        fragsize=2000,
        stride=2000,
        dynamic_stride=True,
        dynamic_stride_threshold=2.0,
    )
    assert indices == [0, 1400]


def test_window_indices_dynamic_exact_multiple():
    # 4000 bp contig -> two windows at 0 and 2000.
    indices = _window_indices(
        seqlen=4000,
        fragsize=2000,
        stride=2000,
        dynamic_stride=True,
        dynamic_stride_threshold=2.0,
    )
    assert indices == [0, 2000]


def test_window_indices_dynamic_long_contig_unchanged():
    # 6000 bp contig (>= 2 * 2000) uses fixed stride.
    indices = _window_indices(
        seqlen=6000,
        fragsize=2000,
        stride=2000,
        dynamic_stride=True,
        dynamic_stride_threshold=2.0,
    )
    assert indices == [0, 2000, 4000]


def test_window_indices_dynamic_3999bp():
    # 3999 bp is short (< 2 * 2000); two evenly distributed windows.
    indices = _window_indices(
        seqlen=3999,
        fragsize=2000,
        stride=2000,
        dynamic_stride=True,
        dynamic_stride_threshold=2.0,
    )
    assert indices == [0, 1999]


def test_fragment_generator_dynamic_stride(tmp_path: Path):
    fasta = tmp_path / "short.fa"
    fasta.write_text(
        f">contig1 len=3400\n{'A' * 3400}\n>contig2 len=2000\n{'C' * 2000}\n"
    )

    # Default behaviour: one window per contig.
    default_frags = list(
        fragment_generator(
            str(fasta),
            fragsize=2000,
            stride=2000,
            num=2,
            no_progress=True,
            dustmask=False,
        )
    )
    assert len(default_frags) == 2

    # Dynamic stride: two windows for the 3400 bp contig.
    dynamic_frags = list(
        fragment_generator(
            str(fasta),
            fragsize=2000,
            stride=2000,
            num=2,
            no_progress=True,
            dustmask=False,
            dynamic_stride=True,
        )
    )
    assert len(dynamic_frags) == 3
    contig1_indices = [
        int(f.split(",")[2]) for f in dynamic_frags if f.split(",")[1] == "contig1"
    ]
    assert contig1_indices == [0, 1400]


def test_fragment_generator_short_contig_whole_window(tmp_path: Path):
    fasta = tmp_path / "mixed.fa"
    fasta.write_text(
        f">long len=3000\n{'A' * 3000}\n"
        f">short len=1500\n{'C' * 1500}\n"
        f">tiny len=900\n{'G' * 900}\n"
    )

    # Two-pass split: long pass only.
    long_frags = list(
        fragment_generator(
            str(fasta),
            fragsize=2000,
            stride=2000,
            num=3,
            no_progress=True,
            dustmask=False,
            min_len=2000,
            max_len=None,
        )
    )
    assert len(long_frags) == 1
    assert long_frags[0].split(",")[1] == "long"

    # Short pass only.
    short_frags = list(
        fragment_generator(
            str(fasta),
            fragsize=2000,
            stride=2000,
            num=3,
            no_progress=True,
            dustmask=False,
            min_len=1000,
            max_len=1999,
        )
    )
    assert len(short_frags) == 1
    parts = short_frags[0].split(",")
    assert parts[1] == "short"
    assert parts[2] == "0"  # index
    assert parts[3] == "1"  # contig_end flag
    assert parts[5] == "1500"  # seqlen


def test_fragment_generator_min_len_below_fsize(tmp_path: Path):
    fasta = tmp_path / "short.fa"
    fasta.write_text(f">short len=1500\n{'C' * 1500}\n")

    frags = list(
        fragment_generator(
            str(fasta),
            fragsize=2000,
            stride=2000,
            num=1,
            no_progress=True,
            dustmask=False,
            min_len=1000,
        )
    )
    assert len(frags) == 1
    assert frags[0].split(",")[1] == "short"
