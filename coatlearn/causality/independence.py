"""Named-column access to causal-learn conditional-independence tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def independence_test(data: pd.DataFrame, x: str, y: str, *, given=(), method="fisherz") -> float:
    """Return the p-value for x independent of y conditional on given columns."""
    from causallearn.utils.cit import CIT

    names = [x, y, *given]
    if len(names) != len(set(names)):
        raise ValueError("x, y and given must name distinct columns")
    values = data.loc[:, names].to_numpy(dtype=float, copy=True)
    if method not in {"mv_fisherz", "chisq", "gsq"}:
        values = np.nan_to_num(values, nan=0.0)
    test = CIT(values, method)
    return float(test(0, 1, list(range(2, len(names)))))
