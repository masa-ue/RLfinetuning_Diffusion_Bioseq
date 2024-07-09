import numpy as np
import torch
import random
import os

def count_kmers(seqs, k=3):
    counts = {}
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            subseq = seq[i : i + k]
            try:
                counts[subseq] += 1
            except KeyError:
                counts[subseq] = 1
    return counts



def seqs_to_one_hot(seqs, alphabet="ACGT", to_upper=True, out_dtype=np.float64):
    """
    Converts a list of strings to one-hot encodings, where the position of 1s is
    ordered by the given alphabet.
    Arguments:
        `seqs`: a list of N strings, where every string is the same length L
        `alphabet`: string of length D containing the alphabet used to do
            the encoding; defaults to "ACGT", so that the position of 1s is
            alphabetical according to "ACGT"
        `to_upper`: if True, convert all bases to upper-case prior to performing
            the encoding
        `out_dtype`: NumPy datatype of the output one-hot sequences; defaults
            to `np.float64` but can be changed (e.g. `np.int8` drastically
            reduces memory usage)
    Returns an N x L x D NumPy array of one-hot encodings, in the same order as
    the input sequences. Any bases that are not in the alphabet will be given an
    encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Get ASCII codes of alphabet in order
    alphabet_codes = np.frombuffer(bytearray(alphabet, "utf8"), dtype=np.int8)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper() + alphabet
    # Add one example of each base, so np.unique doesn't miss indices later

    one_hot_map = np.identity(len(alphabet) + 1)[:, :-1].astype(out_dtype)

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not in the alphabet gets assigned a higher code
    base_vals[~np.isin(base_vals, alphabet_codes)] = np.max(alphabet_codes) + 1

    # Convert the codes into indices, in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds[:-len(alphabet)]].reshape(
        (len(seqs), seq_len, len(alphabet))
    )
