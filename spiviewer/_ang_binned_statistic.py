from collections import namedtuple
from typing import List, Optional, Union, cast

import emcfile as ef
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import binned_statistic

__all__ = ["ang_binned_statistic"]


def _to_1darray(a, exp_l) -> np.ndarray:
    if isinstance(a, ef.PatternsSOne):
        if a.num_pix == exp_l:
            return cast(np.ndarray, a.sum(axis=0))
    elif isinstance(a, np.ndarray):
        if a.size == exp_l:
            return a.flatten()
        if a.ndim == 2 and a.shape[1] == exp_l:
            return a.sum(axis=0)
    raise ValueError(f"Can not convert {a} to a 1D array with length {exp_l}.")


def _get_rs(detector, mask: Optional[List[int]]):
    if isinstance(detector, ef.Detector):
        pix_idx = np.ones(detector.num_pix, bool)
        if mask is not None:
            for m in mask:
                pix_idx &= detector.mask != m
        return np.linalg.norm(detector.coor, axis=1), pix_idx
    if isinstance(detector, np.ndarray):
        if mask is not None:
            raise ValueError("If detector is a `np.ndarray`, mask must be None.")
        if detector.ndim == 1:
            return detector, np.ones(len(detector), bool)
        elif detector.ndim > 1:
            return np.linalg.norm(detector, axis=0), np.ones(detector.shape[1], bool)
        raise ValueError("0 dim array is not acceptable.")


AngBinnedStatisticResult = namedtuple(
    "AngBinnedStatisticResult", ("statistic", "bin_edges", "binnumber", "radius")
)


def _get_binnumber(rs, bin_edges):
    binnumber = np.digitize(rs, bin_edges)
    binnumber[binnumber > len(bin_edges) - 1] = len(bin_edges) - 1
    binnumber[binnumber < 1] = 1
    return binnumber


def _axis1_ang_binned_statistic(patterns, rs, pix_idx, bins=10, statistic="mean"):
    rsp = rs[pix_idx]
    if isinstance(bins, int):
        bin_edges = np.linspace(rsp.min(), rsp.max(), bins + 1)
    else:
        bin_edges = bin
    col = _get_binnumber(rs, bin_edges)
    row = np.arange(len(col))
    if isinstance(patterns, ef.PatternsSOne):
        r = patterns @ coo_matrix(
            (pix_idx.astype("uint8"), (row, col - 1)),
            shape=(patterns.num_pix, len(bin_edges) - 1),
        )
        r = np.asarray(r.todense())
        binnumber = _get_binnumber(rsp, bin_edges)
        if statistic == "sum":
            return AngBinnedStatisticResult(r, bin_edges, binnumber, rs)
        else:
            return AngBinnedStatisticResult(
                r / np.bincount(col - 1, weights=pix_idx), bin_edges, binnumber, rs
            )
    elif isinstance(patterns, np.ndarray):
        raise NotImplementedError()


def ang_binned_statistic(
    patterns: Union[ef.PatternsSOne, np.ndarray],
    detector: Union[ef.Detector, np.ndarray],
    axis=None,
    mask: Optional[List[int]] = None,
    bins=10,
    statistic="mean",
):
    if statistic not in ["sum", "mean", "min", "max"]:
        raise ValueError("statistic can only be 'sum' | 'mean' | 'min' | 'max'")
    rs, pix_idx = _get_rs(detector, mask)
    if axis is None:
        img = _to_1darray(patterns, len(rs))
        ans = binned_statistic(
            rs[pix_idx], img[pix_idx], bins=bins, statistic=statistic
        )
        return AngBinnedStatisticResult(*ans, rs)
    elif axis == 1:
        return _axis1_ang_binned_statistic(
            patterns, rs, pix_idx, bins=bins, statistic=statistic
        )
    raise ValueError("axis can only be None or 1")
