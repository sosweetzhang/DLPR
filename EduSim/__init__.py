# coding: utf-8

from gym.envs.registration import register
from .Envs import *
from .SimOS import train_eval, MetaAgent
from .spaces import *

register(
    id='KSS-v2',
    entry_point='EduSim.Envs:KSSEnv',
)

register(
    id='MBS-EFC-v0',
    entry_point='EduSim.Envs:EFCEnv',
)

register(
    id='MBS-HLR-v0',
    entry_point='EduSim.Envs:HLREnv',
)

register(
    id='MBS-GPL-v0',
    entry_point='EduSim.Envs:GPLEnv',
)

register(
    id='TMS-v1',
    entry_point='EduSim.Envs:TMSEnv',
)
