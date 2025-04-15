from enum import Enum, auto
from typing import Union, Any
from pathlib import Path
import bz2
import gzip
import logging
import lzma
import os
import pyfastx

logger = logging.getLogger("Jaeger")



class Compression(Enum):
    gzip = auto()
    bzip2 = auto()
    xz = auto()
    noncompressed = auto()


def is_compressed(filepath: Union[str, Path]) -> Any:
    """Checks if a file is compressed (gzip, bzip2 or xz)"""
    with open(filepath, "rb") as fin:
        signature = fin.peek(8)[:8]
        if tuple(signature[:2]) == (0x1F, 0x8B):
            return Compression.gzip
        elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
            return Compression.bzip2
        elif tuple(signature[:7]) == (0xFD,
                                      0x37,
                                      0x7A,
                                      0x58,
                                      0x5A,
                                      0x00,
                                      0x00):
            return Compression.xz
        else:
            return Compression.noncompressed


def get_compressed_file_handle(path: Union[str, Path]) -> Any:
    filepath_compression = is_compressed(path)
    if filepath_compression == Compression.gzip:
        return gzip.open(path, "rt")
    elif filepath_compression == Compression.bzip2:
        return bz2.open(path, "rt")
    elif filepath_compression == Compression.xz:
        return lzma.open(path, "rt")
    else:
        return open(path, "r")


def remove_directory_recursively(directory: Union[str, Path]) -> None:
    """recursively remove contents in a directory"""
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
    os.rmdir(directory)

def delete_all_in_directory(root_path: Union[str, Path]) -> None:
    """delete all files and directories in a path"""
    if os.path.exists(root_path):
        for filename in os.listdir(root_path):
            file_path = os.path.join(root_path, filename)
            if os.path.isdir(file_path):
                delete_all_in_directory(file_path)
                os.rmdir(file_path)
            else:
                os.remove(file_path)
        os.rmdir(root_path)

def remove_directory(directory):
    """remove a directory"""
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                remove_directory(file_path)
        os.rmdir(directory)

def dir_path(path:str) -> str:
    """checks path and creates if absent"""
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def check_file_path(path: str) -> str:
    """checks if file is present"""
    if os.path.isfile(path):
        return path
    else:
        raise f"ERROR:{path} is not a valid file"

def validate_fasta_entries(input_file_path: str,
                           min_len: int = 2048) -> Union[int, Exception]:
    num = 0
    gt_min_len = 0
    # try:
    logger.debug("validating fasta file")
    fa = pyfastx.Fasta(input_file_path, build_index=False)
    for seq in fa:
        num += 1
        gt_min_len += 1 if len(seq[1]) >= min_len else 0
    logger.info(f"{gt_min_len}/{num} entries in {input_file_path}")

    if gt_min_len == 0:
        raise Exception(f"all records in {input_file_path} are < {min_len}bp")

    return num

