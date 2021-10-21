#!/usr/bin/env python
'''
This script reads all raw data in JSON and output 2 csv files,
one for training and the other for test.
'''

from pathlib import Path

import glob
import os
import pandas as pd

if __name__ == '__main__':
    input_files = glob.glob('../data/*.jsonlines')
    dfs = []
    for f in input_files:
        df = pd.read_json(f, lines=True)
        df.insert(0, 'title', Path(f).stem)
        dfs.append(df)
    df_combined = pd.concat(dfs)
    print(df_combined.columns)

    # select only records that have significant amount of usefulness votes
    # for now, choose 10 as the selection rule
    df_sel = df_combined[df_combined.num_voted_helpfulness > 10]

    # Split the combined dataframe into
    # training and test samples.
    # Ref: https://www.geeksforgeeks.org/divide-a-pandas-dataframe-randomly-in-a-given-ratio/
    df_test = df_sel.sample(frac=.2, random_state=42)
    df_train = df_sel.drop(df_test.index)

    # prepare the output folder
    out_dir = '../processed_data'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save split dataframes to file.
    df_test.to_csv(os.path.join(out_dir, 'test_meta_inc.csv'), index=False)
    df_train.to_csv(os.path.join(out_dir, 'train_meta_inc.csv'), index=False)
