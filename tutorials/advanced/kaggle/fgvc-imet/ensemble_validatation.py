# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:39:57 2019

@author: XuanCao

Use predict mode of main.py to generate the targets and predictions of all fold. 
The files should be named as prediction_fold0.npy and target_fold0.npy. 
The store directory is '../*/model_name/'

This script will ensemble the predictions of all folds and find the best local
threshold. The result would be saved as 'model_name_valid.h5'. The sample order
is the same as train.csv.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from al_main_noapex import get_score, binarize_prediction

model_name = 'seresnext101'

ROOT = '../output/{}/'.format(model_name)
N_CLASS = 1103

folds = pd.read_csv('folds.csv')

# load data
preds = []
targets = []
ids = []

for i in tqdm(range(5)):
    p_pred = np.load(ROOT + 'prediction_fold{}.npy'.format(i))
    target = np.load(ROOT + 'target_fold{}.npy'.format(i))
    preds.append(p_pred)
    targets.append(target)
    ids.append(folds[folds['fold'] == i]['id'].values)

all_preds = np.vstack(preds)
all_targets = np.vstack(targets)
all_ids = np.concatenate(ids)

# search for best threshold
argsorted = all_preds.argsort(axis=1)
best_score, th = 0, 0
for threshold in tqdm(np.arange(0, 0.5, 0.01)):
    score = get_score(all_targets, 
                      binarize_prediction(all_preds, threshold, argsorted))
    if score > best_score:
        best_score = score
        th = threshold
print('Best score: %.6f, threshold: %.2f' % (best_score, th))

# store results folloing train_data order
df_res = folds[['id']]

df_pred = pd.DataFrame(all_preds)
df_pred['id'] = all_ids

df_final = pd.merge(df_res, df_pred, how='left', on='id')
df_final.set_index('id', inplace=True)
df_final.to_hdf(ROOT + '{}_valid.h5'.format(model_name), 'prob', index_label='id')