# coding: utf-8
import os
import sys
import json
import random
from longling import wf_open
from tqdm import tqdm
from EduSim.utils import get_proj_path
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('IDALPR')] + 'IDALPR')

data_path = f'{get_proj_path()}/data/dataProcess/junyi/student_log_kt_None'
# data_path = os.path.join(get_proj_path(), 'data', 'dataProcess', 'junyi', 'student_log_kt_None')
with open(data_path, 'r', encoding="utf-8") as f:
    datatxt = f.readlines()

dataOff_path = f'{get_proj_path()}/data/dataProcess/junyi/dataOff'
dataRec_path = f'{get_proj_path()}/data/dataProcess/junyi/dataRec'

random.shuffle(datatxt)
with wf_open(dataOff_path) as wf1, wf_open(dataRec_path) as wf2:
    for i, line in tqdm(enumerate(datatxt), 'splitting...', total=len(datatxt)):
        session = json.loads(line)
        if i <= int(len(datatxt)/2):
            print(json.dumps(session), file=wf1)
        else:
            print(json.dumps(session), file=wf2)
print('data split')
