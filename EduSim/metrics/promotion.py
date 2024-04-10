# coding: utf-8

import numpy as np
from longling import as_list as as_list


def as_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    else:
        return np.asarray(as_list(obj))


def promotion_report(initial_score, final_score, full_score=None, path_length=None, average=True, metrics=None,
                     weights=None):
    """

    Parameters
    ----------
    initial_score: float, list or array
    final_score: float, list or array
    full_score: float, list or array
    path_length: int, list or array
    average: bool

    Returns
    -------
    report: dict

        absp:
            absolute promotion =  final_score - initial_score
        absp_rate:
            absolute promotion rate =  \frac{absolute promotion}{path_length}
        relp:
            relative promotion =  \frac{final_score - initial_score}{full_score}
        relp_rate:
            relative promotion rate = \frac{relative promotion}{path_length}
        norm_relp:
            normalized relative promotion = \frac{final_score - initial_score}{full_score - initial_score}
        norm_relp_rate:
            normalized relative promotion rate = \frac{normalized relative promotion}{path_length}

    Examples
    --------
    >>> report = promotion_report(1, 5, 10)
    >>> report["absp"]
    4.0
    >>> report["relp"]
    0.4
    >>> report = promotion_report(1, 5, 10, 0)
    >>> report["relp_rate"]
    0.0
    >>> report = promotion_report(1, 5, 10, 4)
    >>> report["relp_rate"]
    0.1
    >>> report = promotion_report([5, 20], [9, 60], [10, 100], [4, 5], average=False)
    >>> report["absp"]
    [4, 40]
    >>> report["relp"]
    [0.4, 0.4]
    >>> report["norm_relp"]
    [0.8, 0.5]
    """
    metrics = {"absp", "absp_rate", "relp", "relp_rate", "norm_relp", "norm_relp_rate"} if metrics is None else metrics
    ret = {}

    initial_score = as_array(initial_score)
    final_score = as_array(final_score)

    absp = final_score - initial_score

    if weights is not None:
        absp *= as_array(weights)

    if "absp" in metrics:
        ret["absp"] = absp
    if path_length is not None and "absp_rate" in metrics:
        absp_rate = absp / as_array(path_length)
        absp_rate[absp_rate == np.inf] = 0
        ret["absp_rate"] = absp_rate

    if full_score is not None:
        full_score = as_array(full_score)

        if "relp" in metrics:
            relp = absp / full_score
            ret["relp"] = relp

        if path_length is not None and "relp_rate" in metrics:
            relp_rate = absp / (full_score * path_length)
            relp_rate[relp_rate == np.inf] = 0
            ret["relp_rate"] = relp_rate

        if "norm_relp" in metrics:
            ret["norm_relp"] = absp / (full_score - initial_score)
        if path_length is not None and "norm_relp_rate" in metrics:
            norm_relp_rate = absp / ((full_score - initial_score) * path_length)
            norm_relp_rate[norm_relp_rate == np.inf] = 0
            ret["norm_relp_rate"] = norm_relp_rate

    if average:
        return {k: np.average(v) for k, v in ret.items()}
    else:
        return {k: v.tolist() for k, v in ret.items()}
