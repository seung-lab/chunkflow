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

for file_name in tqdm(os.listdir('log'), desc='loading log files'):
    complete_file_name = os.path.join('log/', file_name)
#     print('file name: {}'.format(complete_file_name))
    with open(complete_file_name) as f:
        d = json.load(f)
    timing = d['time_elapsed']

    item = pd.DataFrame({
        'compute_device': d['compute_device'],
        'convnet_inference': timing['convnet_inference'],
        'upload_output': timing['upload_output'],
        'read_image': timing['read_image'],
        'complete_task': timing['complete_task']
    }, index=[0])
    dfs.append(item)

df = pd.concat(dfs)

grouped_df = df.groupby('compute_device')


print('\n\nmax time:')
print(grouped_df.max()) 

print('\n\nmean time:')
print(grouped_df.mean())

