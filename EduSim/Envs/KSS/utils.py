# coding: utf-8

import os
from longling import json_load, path_append, abs_current_dir
from EduSim.Envs.shared.KSS_KES import KS
from EduSim.utils.io_lib import load_ks_from_csv


def load_items(filepath):
    if os.path.exists(filepath):
        return json_load(filepath)
    else:
        return {}


def load_knowledge_structure(filepath):
    knowledge_structure = KS()
    knowledge_structure.add_edges_from([list(map(int, edges)) for edges in load_ks_from_csv(filepath)])
    return knowledge_structure


def load_learning_order(filepath):
    return json_load(filepath)


def load_configuration(filepath):
    return json_load(filepath)


def load_environment_parameters(directory=None):
    if directory is None:
        directory = path_append(abs_current_dir(__file__), "meta_data")
    return {
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "learning_order": load_learning_order(path_append(directory, "learning_order.json")),
        "items": load_items(path_append(directory, "items.json")),
        "know_item": load_items(path_append(directory, "know_item.json"))
    }



