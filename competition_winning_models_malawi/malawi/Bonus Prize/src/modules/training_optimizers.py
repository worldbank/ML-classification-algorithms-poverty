import scipy.optimize as opt
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def combine_models_with_weights(preds, coeffs):
    return (preds * coeffs).sum(axis=1)


def weight_optimizer_generator(y_train, preds):
    def weight_optimizer(coeffs):
        return log_loss(y_train, combine_models_with_weights(preds, coeffs))

    return weight_optimizer


def cross_validate_weight_optimization(X, y, skf):
    val_preds = np.zeros(len(y))
    coeffs = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        opt_weights = opt.minimize(
            weight_optimizer_generator(y_train, X_train),
            x0=np.ones(X_train.shape[1]) / X_train.shape[1],
            constraints=(
                {'type': 'eq', 'fun': lambda x: 1 - sum(x)},
            ),
            bounds=[(0, 1) for i in range(X_train.shape[1])]
        )

        print(opt_weights['fun'])
        coeffs.append(opt_weights['x'])
        val_preds[val_index] = combine_models_with_weights(X_val, opt_weights['x'])

    return val_preds, coeffs


def get_oof_agg(model_test_preds, model_type, res_name, agg='median'):
    test_oof = model_test_preds[
        model_test_preds.columns[
            model_test_preds.columns.str.contains('{}_{}_*'.format(model_type, res_name))
        ]
    ]

    if agg == 'median':
        res = test_oof.median(axis=1)
    elif agg == 'mean':
        res = test_oof.mean(axis=1)
    else:
        raise ValueError('Unknown agg value!')

    return res


def collect_test_oof_preds(model_test_preds, agg='median', res_num=10):
    res = pd.DataFrame()

    for model_type in ['rf', 'nn', 'lgb', 'xgb', 'lr']:
        for res_name in range(res_num):
            r = get_oof_agg(model_test_preds, model_type, res_name, agg=agg)
            r.name = '{}_{}'.format(model_type, res_name)
            res = pd.concat([res, r], axis=1)

    return res


def get_optimized_weighted_preds_for(country_preds_dict, country_code, num_models=5):
    train = country_preds_dict[country_code]['train']
    test = country_preds_dict[country_code]['test']
    y_train = country_preds_dict[country_code]['y_train']

    res_num = train.shape[1] // num_models  # 5 because there are 5 models

    test_rmedian = collect_test_oof_preds(test, agg='median', res_num=res_num)[train.columns]
    test_rmean = collect_test_oof_preds(test, agg='mean', res_num=res_num)[train.columns]

    cv_split = 10
    skf = StratifiedKFold(n_splits=cv_split, shuffle=True, random_state=610)
    optimized_weights_preds, coeffs = cross_validate_weight_optimization(train, y_train, skf)

    print(
        log_loss(y_train, optimized_weights_preds),
        log_loss(y_train, combine_models_with_weights(train, np.mean(coeffs, axis=0)))
    )

    median_preds = combine_models_with_weights(test_rmedian, np.mean(coeffs, axis=0))
    mean_preds = combine_models_with_weights(test_rmean, np.mean(coeffs, axis=0))

    return median_preds, mean_preds, optimized_weights_preds, coeffs
