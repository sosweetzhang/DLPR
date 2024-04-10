# coding: utf-8


STEP = 10
EPISODE = 20
SUMMARY = 30

str2level = {
    "step": STEP,
    "episode": EPISODE,
    "summary": SUMMARY,
}


def as_level(obj):
    if isinstance(obj, int):
        return obj
    else:
        return str2level[obj]
