# coding: utf-8


def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score
    if normalize_factor == 0:
        return 0
    else:
        return delta / normalize_factor
