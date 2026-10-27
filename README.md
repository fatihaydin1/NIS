# Nimble Instance Selection (NIS)
Unsupervised instance selection via conjectural hyperrectangles: https://doi.org/10.1007/s00521-022-07974-z

- `NIS.m`: the MATLAB implementation used in the paper.
- `NIS.py`: a standalone NumPy port that returns the same indices as `NIS.m`.

## scikit-learn compatible version

NIS is also available in [scikit-reduce](https://github.com/fatihaydin1/scikit-reduce), with a `fit_resample` interface that works in imbalanced-learn pipelines:

```bash
pip install scikit-reduce
```

```python
from skreduce import NIS
X_res, y_res = NIS(alpha=1.0).fit_resample(X, y)
```

## Note on earlier versions of NIS.py

Until October 2026, `NIS.py` computed the standard deviation of each row instead of each column and rounded ties to the nearest even number, so its results differed from `NIS.m` and the paper. The current version gives the same results as `NIS.m`.
