
# coding: utf-8

import os

import numpy as np
import pandas as pd

from datetime import datetime
from collections import Counter


# # Start for feature processing

def merge_add_features(train, test):
    
    merge = pd.concat([train.drop('poor', axis=1), test], axis=0)
    df_new = pd.DataFrame(data=merge.id.unique(), columns=['id'])

    cat_ = []
    num_ = []
    for col in train.columns:
        if train[col].dtype in ['int64', 'float64'] and col not in ['id', 'iid']:
            num_.append(col)
        elif train[col].dtype=='O' and col not in ['poor', 'country']:
            cat_.append(col)
    print("Merged table shape: ", merge.shape, 'Categorical features\', number: ', len(cat_), "Numerical features' number: ", len(num_))

    ids = df_new.id.tolist()
    len_ = len(ids)
    print('number of id: ', len_)
    for col in num_:
        df_new[col+'_mean'] = np.NaN
    for idx, id_ in enumerate(merge.id.unique()):
        if idx % 500 == 0:
            print(idx, id_, str(datetime.now()))
        df_new.at[df_new.id==id_, 'family_num'] = merge[merge.id==id_].shape[0]
        for col in num_:
            li = merge[merge.id==id_][col]
            df_new.at[df_new.id==id_, col+'_mean'] = li.mean()
    print("Finish, shape of joined table: ", df_new.shape)
    return df_new

mwii_train = pd.read_csv('../../../data/raw_mwi/mwi_aligned_indiv_train.csv')
mwii_test = pd.read_csv('../../../data/raw_mwi/mwi_aligned_indiv_test.csv')

df_new_mwi = merge_add_features(mwii_train, mwii_test)

df_new_mwi[:len(mwii_train.id.unique())].to_csv('../input/feature_mwi_train_ind.csv', index=False)
df_new_mwi[len(mwii_train.id.unique()):].to_csv('../input/feature_mwi_test_ind.csv', index=False)
