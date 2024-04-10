# coding: utf-8


__all__ = ["extract_relations", "build_json_sequence"]

import sys
import os
from longling import path_append
from EduSim.utils import get_proj_path, get_raw_data_path
from junyi import build_knowledge_graph
from KnowledgeTracing import select_n_most_frequent_students

sys.path.append(os.path.dirname(sys.path[0]))


def extract_relations(src_root: str = "../raw_data/junyi/", tar_root: str = "../data/junyi/"):
    build_knowledge_graph(
        src_root, tar_root,
        ku_dict_path="graph_vertex.json",
        prerequisite_path="prerequisite.json",
        similarity_path="similarity.json",
        difficulty_path="difficulty.json",
    )


def build_json_sequence(src_root: str = "../raw_data/junyi/", tar_root: str = "../data/junyi/",
                        ku_dict_path: str = "../data/junyi/graph_vertex.json", n: int = 1000):
    select_n_most_frequent_students(
        path_append(src_root, "junyi_ProblemLog_for_PSLC.txt", to_str=True),
        path_append(tar_root, "student_log_kt_", to_str=True),
        ku_dict_path,
        n,
    )



def build_demo_data(src,tar):
    with open(src,"r") as f:
        lines = f.readlines()[:4000]
        print("reading...")

    with open(tar,"w") as f:
        f.writelines(lines)
        print("writing...")





if __name__ == "__main__":

    src_root = f'{get_raw_data_path()}/junyi/'
    tar_root = f'{get_proj_path()}/data/dataProcess/junyi/'
    src_demo = f'{get_raw_data_path()}/junyi/junyi_ProblemLog_for_PSLC.txt'
    tar_demo = f'{get_proj_path()}/data/dataDemo/junyi/junyi_ProblemLog_for_PSLC.txt'
    extract_relations(src_root=src_root, tar_root=tar_root)

