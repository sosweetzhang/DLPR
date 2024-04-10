# coding: utf-8

import csv
from longling import as_io


def load_ks_from_csv(edges):
    with as_io(edges) as f:
        for line in csv.reader(f, delimiter=","):
            yield line
