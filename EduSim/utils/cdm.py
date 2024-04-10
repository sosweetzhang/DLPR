# coding: utf-8
__all__ = ["irt"]

import numpy as np
import math

import torch


def irt(ability, difficulty, discrimination=5, c=0.25):
    """
    Examples
    --------
    >>> round(irt(3, 1), 2)
    1.0
    >>> round(irt(1, 5), 3)
    0.25
    """
    return c + (1 - c) / (1 + math.exp(-1.7 * discrimination * (ability - difficulty)))


def dina(abilities, guessing, skipping):
    """
    Examples
    --------
    >>> dina([1, 1, 1], 0, 0)
    1
    >>> dina([0, 0], 1, 0)
    1
    >>> dina([1, 1], 1, 1)
    0
    >>> "%.2f" % dina([0.5, 0.5], 0.2, 0.2)
    '0.28'
    """
    eta = np.prod(abilities)
    return guessing ** (1 - eta) * (1 - skipping) ** eta
