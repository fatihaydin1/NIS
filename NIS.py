"""Nimble Instance Selection (NIS): a NumPy port of NIS.m.

F. Aydin, "Unsupervised instance selection via conjectural hyperrectangles,"
Neural Computing and Applications, 35, 5335-5349, 2023.
https://doi.org/10.1007/s00521-022-07974-z

For a scikit-learn compatible version, see
https://github.com/fatihaydin1/scikit-reduce
"""

import warnings

import numpy as np


def NIS(X, alpha=1.0):
    """Return the indices (0-based) of the instances that NIS keeps.

    The indices are in the same order as those returned by NIS.m.
    """
    X = np.asarray(X, dtype=float)

    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        # Column minima and sample standard deviations, ignoring NaN
        # (MATLAB: min(X,[],1,'omitnan') and std(X,'omitnan')).
        u = np.nanmin(X, axis=0)
        v = np.nanstd(X, axis=0, ddof=1)
        transformedX = alpha * ((X - u) / v)

    # Replace NaN values (missing data, zero-variance columns) with zero.
    transformedX[np.isnan(transformedX)] = 0

    # Round to the nearest integer with ties away from zero, like MATLAB's
    # round (np.round rounds ties to the nearest even number).
    a = np.abs(transformedX)
    r = np.floor(a)
    r += (a - r) >= 0.5
    transformedX = np.copysign(r, transformedX)

    # Keep the first instance of each unique row (MATLAB: unique(..., 'rows', 'first')).
    _, indices = np.unique(transformedX, axis=0, return_index=True)
    return indices


if __name__ == "__main__":
    import sys
    from time import time

    import pandas as pd

    # Usage: python NIS.py dataset.csv [alpha]   (the last column is the class)
    dataset = pd.read_csv(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    X = dataset.iloc[:, :-1].to_numpy()

    t0 = time()
    indices = NIS(X, alpha)
    elapsed_time = time() - t0

    print("Elapsed time:", elapsed_time, sep="\t")
    print("The number of data:", X.shape[0], sep="\t")
    print("The number of unique data:", indices.size, sep="\t", end="\n\n")
    print("Instances have been reduced by {r:8.2f}%".format(r=(X.shape[0] - indices.size) * 100 / X.shape[0]))
