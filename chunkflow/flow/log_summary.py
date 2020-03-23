#!/usr/bin/env python
# coding: utf-8
# Usage:
# download log folder in cloud storage
# run this script to compute summary

import os
import json
import click
import numpy as np
import pandas as pd
from tqdm import tqdm

pd.set_option('precision', 0)

def load_log(log_dir):
    
    collected = dict()

    for file_name in tqdm(os.listdir(log_dir), desc='loading log files'):
        complete_file_name = os.path.join(log_dir, file_name)
        # print('file name: {}'.format(complete_file_name))
        with open(complete_file_name) as f:
            d = json.load(f)

        timing = d['timer']
        
        if not collected:
            # this is the first time, construct the keys
            collected['compute_device'] = []
            collected['complete_task'] = []
            for k in timing.keys():
                collected[k] = []

        time_per_task = 0.0
        for t in d['timer'].values():
            time_per_task += t
        collected['complete_task'].append( time_per_task )

        collected['compute_device'].append( d['compute_device'] )

        for k, v in timing.items():
            collected[k].append( v )

    df = pd.DataFrame.from_dict(collected, orient='columns')
    return df


def print_log_statistics(df, output_size=None):
    grouped_df = df.groupby('compute_device')
    
    print('\n\nmean time (sec):')
    print(grouped_df.mean())

    print('\n\nmax time (sec):')
    print(grouped_df.max())
    
    print('\n\nmin time (sec):')
    print(grouped_df.min())

    if output_size:
        print('\n\nmean speed (mv/s)')
        print(np.prod(output_size) / grouped_df.mean() / 1e3)

    print('\n\nsummation of time (hour):')
    print(grouped_df.sum() / 3600)
