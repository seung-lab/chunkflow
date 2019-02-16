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
        
    if 'device' in d:
        item = pd.DataFrame({
            #'output_bbox': file_name,
            'device': d['device'],
            'inference_time': d['inference_time'],
            'upload_output_time': d['upload_output_time'],
            'read_image_time': d['read_image_time'],
#             'create_output_thumbnail_time': d['create_output_thumbnail'],
            'time_per_task': d['total_time']
        }, index=[0])
    item
    dfs.append(item)

df = pd.concat(dfs)


print('\n\nmax time:')
df_max = df.groupby('device').max()
print(df_max) 

print('\n\nmean time:')
df_mean = df.groupby('device').mean()
print(df_mean)


