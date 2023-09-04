
import numpy as np

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr > 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr > 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def mask2surface(region_mask):
    upperbound = first_nonzero(region_mask, 1, invalid_val=-1)
    bottenbound = last_nonzero(region_mask, 1, invalid_val=-1)
    print(upperbound.shape, bottenbound.shape)
    return upperbound,bottenbound