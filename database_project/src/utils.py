

from typing import List, Iterable
from itertools import tee
import re
import hashlib
import re
import struct
import numpy as np

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
NON_ALPHA = re.compile("[^A-Za-z_0-9]")
def ngrams(sequence: List[str], n: int, min_ngram_size: int = 5) -> Iterable:
    """
    Code taken from NLTK, without padding.

    Parameters
    ----------
    sequence : list
        The sequence of items to be converted into n-grams.
    n : int
        The order of the n-grams to be extracted.
    min_ngram_size : int
        The minimum number of items in the sequence to generate n-grams.

    Returns
    -------
    Iterable
        The n-grams generated from the sequence.

    Examples
    --------
    >>> list(ngrams(['a', 'b', 'c', 'd'], 2))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    >>> list(ngrams(['a', 'b', 'c', 'd'], 3))
    [('a', 'b', 'c'), ('b', 'c', 'd')]
    """
    if len(sequence) < min_ngram_size:
        return []

    iterables = tee(sequence, n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
        The first 4 bytes (32 bits) of the SHA1 hash of the input data.

    Examples
    --------
    >>> sha1_hash32(b"hello")
    499578026
    >>> bin(sha1_hash32(b"hello"))
    '0b11101110001101111010010101010'
    >>> sha1_hash32(b"hello world").bit_length()
    30
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]

from typing import Set

def tokenize(content: str, ngram_size: int, min_ngram_size: int) -> Set[str]:
    tokens = {
        " ".join(t)
        for t in ngrams(NON_ALPHA.split(content), ngram_size, min_ngram_size)
    }
    return tokens


def hash_content(content: str, num_perm: int, ngram_size: int, min_ngram_size: int, permutations: np.ndarray):
    hashvalues = np.ones(num_perm, dtype=np.uint64) * MAX_HASH
    tokens = tokenize(content, ngram_size, min_ngram_size)
    hv = np.array(
        [sha1_hash32(token.encode("utf-8")) for token in tokens], dtype=np.uint64
    )
    a, b = permutations
    phv = np.bitwise_and(
        ((hv * np.tile(a, (len(hv), 1)).T).T + b) % MERSENNE_PRIME, MAX_HASH
    )
    hashvalues = np.vstack([phv, hashvalues]).min(axis=0)
    return hashvalues

