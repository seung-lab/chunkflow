#!/usr/bin/env python
# coding: utf-8
# Usage:
# download log folder in cloud storage
# run this script to compute summary

import os
import json
import pandas as pd
from tqdm import tqdm

dfs = []
skiped_num = 0

for file_name in tqdm(os.listdir('log'), desc='loading log files'):
    complete_file_name = os.path.join('log/', file_name)
#     print('file name: {}'.format(complete_file_name))
    with open(complete_file_name) as f:
        d = json.load(f)

    if 'compute_device' not in d:
        # the mask is all black, most of the operations were skiped
        skiped_num += 1
        continue

    timing = d['timer']
    
    time_per_task = 0.0
    for t in d['timer'].values():
        time_per_task += t
    item = pd.DataFrame({
        'compute_device': d['compute_device'],
        'inference': timing['inference'],
        'save': timing['save'],
        'cutout': timing['cutout'],
        'complete_task': time_per_task,
    }, index=[0])
    dfs.append(item)

df = pd.concat(dfs)

grouped_df = df.groupby('compute_device')

print('\n There are {} tasks that have black mask and most of operations were skipped.'.format(skiped_num))

print('\n\nmax time:')
print(grouped_df.max()) 

print('\n\nmean time:')
print(grouped_df.mean())

