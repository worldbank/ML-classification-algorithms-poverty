import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb

from xgboost import XGBClassifier
from joblib import Parallel, delayed
import logging


def infer(model_id, res_name, X_infer, fill_type, classifiers):
    infer_preds = []

    for model in classifiers:
        if model_id == 'xgb':
            tpreds = model.predict_proba(X_infer, ntree_limit=model.best_ntree_limit)[:, 1]
        elif model_id == 'lgb':
            tpreds = model.predict_proba(X_infer, num_iteration=model.best_iteration_)[:, 1]
        else:
            tpreds = model.predict_proba(X_infer)[:, 1]

        infer_preds.append(tpreds)

    cv_split = len(classifiers)

    infer_preds = pd.DataFrame(
        np.vstack(infer_preds).T,
        index=X_infer.index,
        columns=['{}_{}_{}'.format(model_id, res_name, i) for i in range(cv_split)]
    )

    return {
        '{}_{}'.format(model_id, res_name): infer_preds,
    }


def cv_train_model(
    X_train, y_train, X_opt_val, y_opt_val,
    params, model_id, res_name, cv_func,
    fill_type, cv_split=10
):
    skf = StratifiedKFold(n_splits=cv_split, shuffle=True, random_state=1029)
    y_oof_preds, classifiers = cv_func(X_train, y_train, params, skf)

    train_val_loss = log_loss(y_train, y_oof_preds)

    opt_val_preds = []

    for model in classifiers:
        if model_id == 'xgb':
            ovpreds = model.predict_proba(X_opt_val, ntree_limit=model.best_ntree_limit)[:, 1]
        elif model_id == 'lgb':
            ovpreds = model.predict_proba(X_opt_val, num_iteration=model.best_iteration_)[:, 1]
        else:
            ovpreds = model.predict_proba(X_opt_val)[:, 1]

        opt_val_preds.append(ovpreds)

    opt_val_preds = pd.DataFrame(
        np.vstack(opt_val_preds).T,
        index=X_opt_val.index,
        columns=['{}_{}_{}'.format(model_id, res_name, i) for i in range(cv_split)]
    )
    opt_val_loss = log_loss(y_opt_val, opt_val_preds.mean(axis=1))

    if not X_opt_val.empty:
        logging.info('{} loss: {}'.format(model_id, train_val_loss))

    return y_oof_preds, classifiers, opt_val_loss


def train_lr(X_train, y_train, X_val, val_index, params, ix):
    model = LogisticRegression(**params)
    model.fit(
        X_train, y_train
    )

    preds = model.predict_proba(X_val)
    val_preds = preds[:, 1]

    return [model, val_preds, val_index, params, ix]


def train_nn_keras(X_train, y_train, X_val, val_index, params, ix):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    X_train_train, X_test_test, y_train_train, y_test_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    model = Sequential()
    model.add(Dense(10, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    es_callback = keras.callbacks.EarlyStopping(patience=2)
    callbacks = [es_callback]

    model.fit(
        X_train_train, y_train_train, verbose=0,
        batch_size=1024, epochs=1000,
        validation_data=(X_test_test, y_test_test),
        callbacks=callbacks
    )

    preds = model.predict(X_val).ravel()
    val_preds = preds

    return [model, val_preds, val_index, params, ix]


def train_nn_sklearn(X_train, y_train, X_val, val_index, params, ix):
    model = MLPClassifier(**params)
    model.fit(
        X_train, y_train
    )

    preds = model.predict_proba(X_val)
    val_preds = preds[:, 1]

    return [model, val_preds, val_index, params, ix]


def lr_cv(X, y, params, skf, return_classifiers=True):
    y_oof_preds = np.zeros(len(y))
    lr_classifiers = []

    model = LogisticRegression(**params)

    if not return_classifiers:
        y_oof_preds = cross_val_predict(model, X, y, cv=skf, method='predict_proba', n_jobs=7)[:, 1]

    else:
        res = Parallel(n_jobs=7)(
            delayed(train_lr)(
                X.iloc[train_index], y[train_index],
                X.iloc[val_index], val_index,
                params, ix
            ) for ix, (train_index, val_index) in enumerate(skf.split(X, y))
        )

        for model, val_preds, val_index, params, ix in res:
            lr_classifiers.append(model)
            y_oof_preds[val_index] = val_preds

    return y_oof_preds, lr_classifiers


def nn_cv(X, y, params=None, skf=None, use_sklearn=True):
    y_oof_preds = np.zeros(len(y))
    nn_classifiers = []

    if params is None:
        params = {
            'hidden_layer_sizes': (50, 20),
            'alpha': 0.01,
            'max_iter': 1000,
            'early_stopping': True,
            'random_state': 1029
        }

    if use_sklearn:
        model_nn = train_nn_sklearn
        n_jobs = 7
    else:
        model_nn = train_nn_keras
        n_jobs = 1

    res = Parallel(n_jobs=n_jobs)(
        delayed(model_nn)(
            X.iloc[train_index], y[train_index],
            X.iloc[val_index], val_index,
            params, ix
        ) for ix, (train_index, val_index) in enumerate(skf.split(X, y))
    )

    for model, val_preds, val_index, params, ix in res:
        nn_classifiers.append(model)
        y_oof_preds[val_index] = val_preds

    return y_oof_preds, nn_classifiers


def rf_cv(X, y, params, skf):
    y_oof_preds = np.zeros(len(y))
    rf_classifiers = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_train, X_test_test, y_train_train, y_test_test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        model = RandomForestClassifier(**params)
        model.fit(X_train_train, y_train_train)

        rf_classifiers.append(model)

        preds = model.predict_proba(X_val)
        y_oof_preds[val_index] = preds[:, 1]

    return y_oof_preds, rf_classifiers


def lgb_cv(X, y, params, skf):
    y_oof_preds = np.zeros(len(y))
    lgb_classifiers = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_train, X_test_test, y_train_train, y_test_test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        categorical_feature = list(X_train_train.columns[X_train_train.columns.str.startswith('cat_')])

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_train, y_train_train,
            eval_metric='logloss',
            eval_set=[(X_test_test, y_test_test)],
            early_stopping_rounds=20,
            verbose=False,
            categorical_feature=categorical_feature if (len(categorical_feature) > 0) else 'auto'
        )

        lgb_classifiers.append(model)

        preds = model.predict_proba(X_val, num_iteration=model.best_iteration_)
        y_oof_preds[val_index] = preds[:, 1]

    return y_oof_preds, lgb_classifiers


def xgb_cv(X, y, params, skf):
    y_oof_preds = np.zeros(len(y))
    xgb_classifiers = []

    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        X_train_train, X_test_test, y_train_train, y_test_test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        model = XGBClassifier(**params)
        model.fit(
            X_train_train, y_train_train,
            eval_metric='logloss',
            eval_set=[(X_test_test, y_test_test)],
            early_stopping_rounds=20,
            verbose=False
        )

        xgb_classifiers.append(model)

        preds = model.predict_proba(X_val, ntree_limit=model.best_ntree_limit)
        y_oof_preds[val_index] = preds[:, 1]

    return y_oof_preds, xgb_classifiers
