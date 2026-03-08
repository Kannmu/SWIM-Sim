from __future__ import annotations

import itertools
import numpy as np


def standardize_scores(score_dict):
    methods = list(score_dict.keys())
    values = np.asarray([score_dict[m] for m in methods], dtype=np.float64)
    mu = float(values.mean())
    sigma = float(values.std())
    if sigma <= 1e-12:
        sigma = 1.0
    z = (values - mu) / sigma
    return {m: float(v) for m, v in zip(methods, z)}


def pairwise_preferences(score_dict, logistic_scale=1.0, standardize=True):
    scores = standardize_scores(score_dict) if standardize else dict(score_dict)
    out = []
    methods = list(scores.keys())
    for a, b in itertools.combinations(methods, 2):
        delta = (scores[a] - scores[b]) * float(logistic_scale)
        preference_index = 1.0 / (1.0 + np.exp(-delta))
        out.append({"A": a, "B": b, "preference_index": float(preference_index)})
    return scores, out
