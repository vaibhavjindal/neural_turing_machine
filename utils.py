""" Some complex utility functions""" 
import numpy as np
import torch
import torch.nn.functional as F


def _split_cols(mat, lengths):
    """
    Split a 2D matrix to variable length columns.
    Taken from https://github.com/loudinthecloud/pytorch-ntm
    """
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results

def _convolve(w, s):
    """
    Circular convolution implementation.
    Taken from https://github.com/loudinthecloud/pytorch-ntm
    """
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c
