import numpy as np
import pandas as pd
from contextlib import contextmanager
import time
import logging


@contextmanager
def timing(name):
    t0 = time.time()
    yield
    
    log_out = 'Fragment [{}] done in {:.2f} s\n'.format(name, time.time() - t0)
    print(log_out)
    logging.info(log_out)


def get_indiv_important_cols(indiv_train, indiv_cat_train, country_code, min_corr_val=0.05):
    indiv_cat_train[country_code] = (1 * indiv_train.reset_index('id').groupby('id')['poor'].mean())

    important_cols_indiv = indiv_cat_train.astype(float).drop(
        country_code, axis=1
    ).corrwith(
        indiv_cat_train[country_code]
    ).abs().between(min_corr_val, 1)

    return important_cols_indiv[important_cols_indiv == True].index


def round_float_to(number, round_to=0.05):
    return round(number / round_to) * round_to


def get_round_num(num, round_num):
    return int((round_num * (num // round_num)) + round_num)


def get_opt_val_seeds(size, seed=1030):
    np.random.seed(seed)
    opt_val_seeds = np.random.choice([42, 1029, 610, 514], size=size, replace=True)
    
    return opt_val_seeds


def make_country_sub(preds, test_feat, country):
    country_sub = pd.DataFrame(
        data=preds,
        columns=['poor'],
        index=test_feat.index
    )

    country_sub['country'] = country
    return country_sub[['country', 'poor']]
