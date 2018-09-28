
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
a_train = pd.read_csv('../input/A_hhold_train.csv', index_col='id')
b_train = pd.read_csv('../input/B_hhold_train.csv', index_col='id')
c_train = pd.read_csv('../input/C_hhold_train.csv', index_col='id')
# load test data
a_test_o = pd.read_csv('../input/A_hhold_test.csv', index_col='id')
b_test_o = pd.read_csv('../input/B_hhold_test.csv', index_col='id')
c_test_o = pd.read_csv('../input/C_hhold_test.csv', index_col='id')

a_feature_1 = pd.read_csv('../input/feature_a_train_ind.csv', index_col=['id'])
b_feature_1 = pd.read_csv('../input/feature_b_train_ind.csv', index_col=['id'])
c_feature_1 = pd.read_csv('../input/feature_c_train_ind.csv', index_col=['id'])

a_feature_1_test = pd.read_csv('../input/feature_a_test_ind.csv', index_col=['id'])
b_feature_1_test = pd.read_csv('../input/feature_b_test_ind.csv', index_col=['id'])
c_feature_1_test = pd.read_csv('../input/feature_c_test_ind.csv', index_col=['id'])

print('Shape of hhold data A, B, C, Train and test: \n', a_train.shape, b_train.shape, c_train.shape, a_test_o.shape, b_test_o.shape, c_test_o.shape)
print('Used features\' shape from individuals for A, B, C (train, test): \n', a_feature_1.shape, b_feature_1.shape, c_feature_1.shape, a_feature_1_test.shape, b_feature_1_test.shape, c_feature_1_test.shape)
print('Mean value of "poor" for A, B, C: \n',a_train.poor.mean(), b_train.poor.mean(), c_train.poor.mean())

# join *_hhold data with the data from individual data data tables
a_used_features = [col for col in a_feature_1.columns if '_mean' in col or col == 'family_num']
b_used_features = [col for col in b_feature_1.columns if '_mean' in col or col == 'family_num']
c_used_features = [col for col in c_feature_1.columns if col == 'family_num']

a_feature_1 = a_feature_1[a_used_features]
b_feature_1 = b_feature_1[b_used_features]
c_feature_1 = c_feature_1[c_used_features]

a_feature_1_test = a_feature_1_test[a_used_features]
b_feature_1_test = b_feature_1_test[b_used_features]
c_feature_1_test = c_feature_1_test[c_used_features]

a_train = a_train.join(a_feature_1, how='inner')
b_train = b_train.join(b_feature_1, how='inner')
c_train = c_train.join(c_feature_1, how='inner')

a_test_o = a_test_o.join(a_feature_1_test, how='inner')
b_test_o = b_test_o.join(b_feature_1_test, how='inner')
c_test_o = c_test_o.join(c_feature_1_test, how='inner')
print(a_train.shape, b_train.shape, c_train.shape)

num_cols_a = [col for col in a_train.columns if a_train[col].dtype in ['int64', 'float64'] and col not in ['id', 'family_num'] and '_mean' not in col]
# print(num_cols_a)
for col in num_cols_a:
    a_train[col + '_ave'] = a_train[col]/a_train['family_num']
    a_test_o[col + '_ave'] = a_test_o[col]/a_test_o['family_num']
    
# a_hhold = pd.concat([a_train, a_test_o], axis=0)
num_cols_b = [col for col in b_train.columns if b_train[col].dtype in ['int64', 'float64'] and col not in ['id', 'family_num'] and '_mean' not in col]
for col in num_cols_b:
    b_train[col + '_ave'] = b_train[col]/b_train['family_num']
    b_test_o[col + '_ave'] = b_test_o[col]/b_test_o['family_num']
    
num_cols_c = [col for col in c_train.columns if c_train[col].dtype in ['int64', 'float64'] and col not in ['id', 'family_num'] and '_mean' not in col]

for col in num_cols_c:
    c_train[col + '_ave'] = c_train[col]/c_train['family_num']
    c_test_o[col + '_ave'] = c_test_o[col]/c_test_o['family_num']
a_hhold = pd.concat([a_train.drop(['country'], axis=1), a_test_o], axis=0)
# a_hhold = pd.concat([a_train, a_test_o], axis=0)
b_hhold = pd.concat([b_train.drop(['country'], axis=1), b_test_o], axis=0)
c_hhold = pd.concat([c_train.drop(['country'], axis=1), c_test_o], axis=0)
print(a_hhold.shape, b_hhold.shape, c_hhold.shape)
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

a_hhold_p = pre_process_data(a_hhold, nn=True)
print()
b_hhold_p = pre_process_data(b_hhold, nn=True)
print()
c_hhold_p = pre_process_data(c_hhold, nn=True)

aX_train = a_hhold_p.drop('poor', axis=1)[:a_train.shape[0]]
ay_train = np.ravel(a_hhold_p.poor)[:a_train.shape[0]]
a_test = a_hhold_p.drop('poor', axis=1)[a_train.shape[0]:]

bX_train = b_hhold_p.drop('poor', axis=1)[:b_train.shape[0]]
by_train = np.ravel(b_hhold_p.poor)[:b_train.shape[0]]
b_test = b_hhold_p.drop('poor', axis=1)[b_train.shape[0]:]

cX_train = c_hhold_p.drop('poor', axis=1)[:c_train.shape[0]]
cy_train = np.ravel(c_hhold_p.poor)[:c_train.shape[0]]
c_test = c_hhold_p.drop('poor', axis=1)[c_train.shape[0]:]

# # # Read in data
# a_train = pd.read_csv('../input/A_train.csv', index_col=['id'])
# b_train = pd.read_csv('../input/B_train.csv', index_col=['id'])
# c_train = pd.read_csv('../input/C_train.csv', index_col=['id'])
# print(a_train.shape, b_train.shape, c_train.shape)

# aX_train = a_train[list(set(a_train.columns.tolist()) - set(['poor']))]
# bX_train = b_train[list(set(b_train.columns.tolist()) - set(['poor']))]
# cX_train = c_train[list(set(c_train.columns.tolist()) - set(['poor']))]

# ay_train = a_train['poor'].values
# by_train = b_train['poor'].values
# cy_train = c_train['poor'].values

# print(aX_train.shape, bX_train.shape, cX_train.shape)

# a_test = pd.read_csv('../input/A_test.csv', index_col=['id'])
# b_test = pd.read_csv('../input/B_test.csv', index_col=['id'])
# c_test = pd.read_csv('../input/C_test.csv', index_col=['id'])
# print(a_test.shape, b_test.shape, c_test.shape)




# # Start training

paras_a = {
    'splits': 20,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.75,
        'colsample_by_tree': 0.07,
        'verbos_': 1000,
        'col_names': aX_train.columns.tolist(),
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

a_preds, a_loss = train_model(aX_train, ay_train, paras_a, test_ = a_test)

paras_b = {
    'splits': 20,
    'lgb': {
        'max_depth': 3,
        'lr': 0.01,
        'hess': 3.,
        'feature_fraction': 0.025,
        'verbos_': 1000,
        'col_names': bX_train.columns.tolist(),
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 3,
        'subsample': 0.45,
        'colsample_by_tree': 0.03,
        'verbos_': 1000,
        'col_names': bX_train.columns.tolist(),
    },
    'use_nn': True,
    'nn': {
        'nn_l1': 400,
        'nn_l2': 400,
        'epochs': 30,
        'batch': 32,
        'dp': 0.25,
    },
    'w_xgb': 0.4,
    'w_lgb': 0.3,
    'w_nn': 0.3,
}

b_preds, b_loss = train_model(bX_train, by_train, paras_b, test_ = b_test)

used_features_c = [col for col in cX_train.columns if '_mean' not in col]
paras_c = {
    'splits': 10,
    'lgb': {
        'max_depth': 4,
        'lr': 0.01,
        'hess': 1.,
        'feature_fraction': 0.99,
        'verbos_': 1000,
        'col_names': used_features_c,
    },
    'xgb': {
        'eta': 0.01,
        'max_depth': 6,
        'subsample': 0.75,
        'colsample_by_tree': 0.75,
        'verbos_': 1000,
        'col_names': used_features_c,
    },
    'use_nn': False,
    'nn': {
        'nn_l1': 400,
        'nn_l2': 400,
        'epochs': 30,
        'batch': 32,
        'dp': 0.25,
    },
    'w_xgb': 0.7,
    'w_lgb': 0.3,
#     'w_nn': 0.3,
}

used_features_c = [col for col in cX_train.columns if '_mean' not in col]
c_preds, c_loss = train_model(cX_train[used_features_c], cy_train, paras_c, test_ = c_test[used_features_c])

def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')
submission = pd.concat([a_sub, b_sub, c_sub])

print(a_sub.poor.agg(['max', 'min', 'mean', 'median']))
print(b_sub.poor.agg(['max', 'min', 'mean', 'median']))
print(c_sub.poor.agg(['max', 'min', 'mean', 'median']))

# lgb (new) 10 folds --------------------new
log_loss_mean = a_loss*(4041/8832) + b_loss*(1604/8832) + c_loss*(3187/8832)
print('Local logloss cross validation score: {}'.format(log_loss_mean))


submission.to_csv('../subs/sub_logloss_'+str(log_loss_mean)+'.csv')

