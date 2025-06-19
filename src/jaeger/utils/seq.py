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
    return "".join(complement_dict.get(base, "N") for base in reversed(dna_sequence))
