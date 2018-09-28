
# coding: utf-8

import os
from datetime import datetime

import numpy as np
# np.random.seed(123)

import pandas as pd
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras import models

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import math
import lightgbm as lgb
import xgboost as xgb

from collections import Counter
from models import *

# prepare data
DATA_DIR = os.path.join('..', 'input')

# load *_hhold training data
mwi_train = pd.read_csv('../../../data/raw_mwi/mwi_aligned_hhold_train.csv', index_col='id')
# load test data
mwi_test_o = pd.read_csv('../../../data/raw_mwi/mwi_aligned_hhold_test.csv', index_col='id')

mwi_feature_1 = pd.read_csv('../input/feature_mwi_train_ind.csv', index_col=['id'])

mwi_feature_1_test = pd.read_csv('../input/feature_mwi_test_ind.csv', index_col=['id'])

print('Shape of hhold data mwi Train and test: \n', mwi_train.shape, mwi_test_o.shape)
print('Used features\' shape from individuals for mwi (train, test): \n', mwi_feature_1.shape, mwi_feature_1_test.shape)
print('Mean value of "poor" for mwi: \n', mwi_train.poor.mean())

# join *_hhold data with the data from individual data data tables
mwi_used_features = [col for col in mwi_feature_1.columns if '_mean' in col or col == 'family_num']

mwi_feature_1 = mwi_feature_1[mwi_used_features]

mwi_feature_1_test = mwi_feature_1_test[mwi_used_features]

mwi_train = mwi_train.join(mwi_feature_1, how='inner')

mwi_test_o = mwi_test_o.join(mwi_feature_1_test, how='inner')
print(mwi_train.shape)

num_cols_mwi = [col for col in mwi_train.columns if mwi_train[col].dtype in ['int64', 'float64'] and col not in ['id', 'family_num'] and '_mean' not in col]
# print(num_cols_mwi)
for col in num_cols_mwi:
    mwi_train[col + '_ave'] = mwi_train[col]/mwi_train['family_num']
    mwi_test_o[col + '_ave'] = mwi_test_o[col]/mwi_test_o['family_num']

mwi_hhold = pd.concat([mwi_train.drop(['country'], axis=1), mwi_test_o], axis=0)
print(mwi_hhold.shape)

# Standardize features
def standardize(df, numeric_only=True):
    
    numeric = df.select_dtypes(include=['int64', 'float64'])
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df

# def encode_cat(df):
#     for col in df.columns:
#         if df[col].dtype not in ['int64', 'float64', 'bool']:
#             len_ = df[col].unique()
#             dict_ = {cl: i for i, cl in enumerate(df[col].unique())}
# #             df[col] = df[col].astype('category')
#             df[col] = df[col].apply(lambda x: dict_[x])
#     return df
    
def pre_process_data(df, nn=False):
    print("Input shape:\t{}".format(df.shape))
    df = standardize(df)
#     print("After standardization {}".format(df.shape))
    if nn:
        current_cols = df.columns
#         print('poor' in df.columns)
        poor_ = df.poor.values
        df = pd.get_dummies(df.drop('poor', axis=1), drop_first=True)
        df['poor'] = poor_.astype('bool')
        all_nan_cols = []
        for col in df.columns:
            if df[col].isnull().sum() == df.shape[0]:
                all_nan_cols.append(col)
        df.drop(all_nan_cols, axis=1, inplace=True)
        df.fillna(df.median(), inplace=True)
    else:
        df = encode_cat(df)
    print('Final shape {}'.format(df.shape))
    return df

mwi_hhold_p = pre_process_data(mwi_hhold, nn=True)
print()

mwiX_train = mwi_hhold_p.drop('poor', axis=1)[:mwi_train.shape[0]]
mwiy_train = np.ravel(mwi_hhold_p.poor)[:mwi_train.shape[0]]
mwi_test = mwi_hhold_p.drop('poor', axis=1)[mwi_train.shape[0]:]

# # # Read in data
# a_train = pd.read_csv('../input/A_train.csv', index_col=['id'])
# print(a_train.shape)

# aX_train = a_train[list(set(a_train.columns.tolist()) - set(['poor']))]

# ay_train = a_train['poor'].values

# print(aX_train.shape)

# a_test = pd.read_csv('../input/A_test.csv', index_col=['id'])
# print(a_test.shape)


# # Start training

paras_mwi = {
    'splits': 20,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.07,
        'verbos_': 1000,
        'col_names': mwiX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_by_tree': 0.07,
        'verbos_': 1000,
        'col_names': mwiX_train.columns.tolist(),
    },
    'use_nn': True,
    'nn': {
        'nn_l1': 300,
        'nn_l2': 300,
        'epochs': 75,
        'batch': 64,
        'dp': 0.,
    },
    'w_xgb': 0.45,
    'w_lgb': 0.25,
    'w_nn': 0.3,
}

mwi_preds, mwi_loss = train_model(mwiX_train, mwiy_train, paras_mwi, test_ = mwi_test)

def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['mwi']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
mwi_sub = make_country_sub(mwi_preds, mwi_test, 'mwi')
submission = mwi_sub

print(mwi_sub.poor.agg(['max', 'min', 'mean', 'median']))

# lgb (new) 10 folds --------------------new
log_loss_mean = mwi_loss
print('Local logloss cross validation score: {}'.format(log_loss_mean))

submission.to_csv('../subs/sub_logloss_'+str(log_loss_mean)+'.csv')