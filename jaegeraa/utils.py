from typing import Iterable, Any, Tuple
from enum import Enum, auto
import bz2
import gzip
import logging
import lzma
import os

class Compression(Enum):
    gzip = auto()
    bzip2 = auto()
    xz = auto()
    noncompressed = auto()
    
def is_compressed(filepath):
    """ Checks if a file is compressed (gzip, bzip2 or xz) """
    with open(filepath, "rb") as fin:
        signature = fin.peek(8)[:8]
        if tuple(signature[:2]) == (0x1F, 0x8B):
            return Compression.gzip
        elif tuple(signature[:3]) == (0x42, 0x5A, 0x68):
            return Compression.bzip2
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            return Compression.xz
        else:
            return Compression.noncompressed
def get_compressed_file_handle(path):
    filepath_compression = is_compressed(path)
    if filepath_compression == Compression.gzip:
        f = gzip.open(path, "rt")
    elif filepath_compression == Compression.bzip2:
        f = bz2.open(path, "rt")
    elif filepath_compression == Compression.xz:
        f = lzma.open(path, "rt")
    else:
        f = open(path, "r")
    return f

def signal_fl(it):
    '''get a signal at the begining and the end of a iterator'''
    iterable = iter(it)
    yield True, next(iterable)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var
def signal_l(it):
    '''get a signal at the end of the iterator'''
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield 0, ret_var
        ret_var = val
    yield 1, ret_var

def get_num_entries(file_handle):
  '''returns the number of entries in a fasta file'''
  pass

def is_low_quality(sequence):
  '''returns True if more than 30% of the input sequence/fragment is covered with ambigious nucleotides 
     predictions made on such regions will be masked in the final prediction
  '''
  pass 
