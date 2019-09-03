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


def load_log(log_dir):
    dfs = []
    skiped_num = 0

    for file_name in tqdm(os.listdir(log_dir), desc='loading log files'):
        complete_file_name = os.path.join(log_dir, file_name)
        # print('file name: {}'.format(complete_file_name))
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
        item = pd.DataFrame(
            {
                'compute_device': d['compute_device'],
                'inference': timing['inference'],
                'save': timing['save'],
                'cutout': timing['cutout'],
                'complete_task': time_per_task,
            },
            index=[0])
        dfs.append(item)

    df = pd.concat(dfs)
    print(
        '\n There are {} tasks that have black mask and most of operations were skipped.'
        .format(skiped_num))
    return df


def print_log_statistics(df, output_size=None):
    grouped_df = df.groupby('compute_device')

    print('\n\ngrouped dataframe')
    print(grouped_df)

    print('\n\nmax time (sec):')
    print(grouped_df.max())

    print('\n\nmean time (sec):')
    print(grouped_df.mean())

    if output_size:
        print('\n\nmean speed (mv/s)')
        print(np.prod(output_size) / grouped_df.mean() / 1e3)

    print('\n\nsummation of time (hour):')
    print(grouped_df.sum() / 3600)