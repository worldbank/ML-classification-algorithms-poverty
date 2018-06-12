import os
import sys

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../src")

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold

from . import config
from .metrics import gini_norm, f1
from .DataReader import FeatureDictionary, DataParser
from .DeepFM import DeepFM
from data.load_data import (get_country_filepaths, 
                            split_features_labels_weights, 
                            load_data)
from evaluation import calculate_metrics, evaluate_model

ALGORITHM_NAME = 'deepfm'
# TRAIN_PATH, TEST_PATH, QUESTIONS_PATH = get_country_filepaths(COUNTRY)

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def load_data_for_deepfm(country, test_only=False, undersample=False, oversample=False, overunder=False):
    
    TRAIN_PATH, TEST_PATH, QUESTIONS_PATH = get_country_filepaths(country)
    
    dfTest, y_test, w_test = load_data(TEST_PATH)
    dfTest["target"] = y_test
    # rename "hid" --> "id" for data parser
    dfTest.index.name = "id"
    dfTest = dfTest.reset_index()
    
    # get column names
    cols = [c for c in dfTest.columns if c not in ["id", "target"]]
    # columns to ignore
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]
    
    if not test_only:
        dfTrain, y_train, w_train = load_data(TRAIN_PATH)
    
        if undersample or oversample or overunder:
            print("X shape before resampling: ", dfTrain.shape)
        if undersample:
            dfTrain, y_train = RandomUnderSampler().fit_sample(dfTrain, y_train)
            dfTrain = pd.DataFrame(data=dfTrain, columns=cols)
            print("X shape after undersampling: ", dfTrain.shape)
        if oversample:
            dfTrain, y_train = SMOTE().fit_sample(dfTrain, y_train)
            dfTrain = pd.DataFrame(data=dfTrain, columns=cols)
            print("X shape after oversampling: ", dfTrain.shape)
        if overunder:
            dfTrain, y_train = SMOTEENN().fit_sample(dfTrain, y_train)
            dfTrain = pd.DataFrame(data=dfTrain, columns=cols)
            print("X shape after SMOTEENN: ", dfTrain.shape)

        # rename label "poor" --> "target"
        # also combine label into features df for data parser
        dfTrain["target"] = y_train
        dfTrain.index.name = "id"
        dfTrain = dfTrain.reset_index()
    
    def preprocess(df):
        cols = [c for c in df.columns if c not in ["id", "target"]]
        return df

    if not test_only:
        dfTrain = preprocess(dfTrain)
        X_train = dfTrain[cols].values
        y_train = dfTrain["target"].values
    
    dfTest = preprocess(dfTest)
    X_test = dfTest[cols].values
    ids_test = dfTest["id"].values
    
    # get col index if categorical
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    if not test_only:
        return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices
    else:
        return dfTest, X_test, ids_test, cat_features_indices


def run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    
    # create feature dictionary based on raw df
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    
    # instantiate data parser
    data_parser = DataParser(feat_dict=fd)
    
    # parse data
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    # empty data?
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    
    # to get values from indices
    _get = lambda x, l: [x[i] for i in l]
    
    # preallocate gini results
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    
    # fit
    for i, (train_idx, valid_idx) in enumerate(folds):
        
        # split
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)
        
        # fit on fold
        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        # predict on fold
        y_train_meta[valid_idx,0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        
        # score on validation
        #gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_cv[i] = f1(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    return y_train_meta, y_test_meta, dfm


def _make_submission(ids, y_pred, filename="submission.csv"):
    if not os.path.exists(config.SUB_DIR):
        os.makedirs(config.SUB_DIR)
        
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(config.SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green", "black", "yellow"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("%s"%model_name)
    plt.legend(legends)
    
    if not os.path.exists(config.FIG_DIR):
        os.makedirs(config.FIG_DIR)
    
    plt.savefig(os.path.join(config.FIG_DIR, model_name + ".png"))    
    plt.close()

if __name__ == "__main__":
    
    # load data
    dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()


    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(X_train, y_train))


    # ------------------ DeepFM Model ------------------
    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [64, 32, 32, 16, 8],
        "dropout_deep": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 50,
        "batch_size": 1024,
        "learning_rate": 0.01,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.895,
        "l2_reg": 0.001,
        "verbose": True,
        "eval_metric": f1,
        "random_seed": config.RANDOM_SEED
    }


    y_train_dfm, y_test_dfm, dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

    # get train preds
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)

    # this should give preds for all training (see inside _run_base_model)
    y_train_preds = dfm.predict(Xi_train, Xv_train)

    # get test preds
    y_test = dfTest['target'].values.astype(int)
    y_pred = np.round(y_test_dfm)
    y_prob = y_test_dfm

    ##### manual pov error pred ###### so much shame ### 
    # Recombine the entire dataset to get the actual poverty rate
    X_train, y_train_, w_train = load_data(TRAIN_PATH,
                                          standardize_columns='numeric',
                                          ravel=True,
                                          selected_columns=None)
    X_test, y_test_, w_test = load_data(TEST_PATH,
                                       standardize_columns='numeric',
                                       ravel=True,
                                       selected_columns=None)
    pov_rate = pd.DataFrame(np.vstack((np.vstack((y_train_, w_train)).T,
                                       np.vstack((y_test_, w_test)).T)),
                            columns=['poor', 'wta_pop'])
    pov_rate_actual = (pov_rate.wta_pop * pov_rate.poor).sum() / pov_rate.wta_pop.sum()

    # Make predictions on entire dataset to get the predicted poverty rate
    pov_rate['pred'] = np.concatenate((y_train_preds.ravel(), y_prob.ravel()))
    pov_rate_pred = (pov_rate.wta_pop * pov_rate.pred).sum() / pov_rate.wta_pop.sum()

    print("Actual poverty rate: {:0.2%} ".format(pov_rate_actual))
    print("Predicted poverty rate: {:0.2%} ".format(pov_rate_pred))

    # sloppy save
    X_train, y_train_, w_train = load_data(TRAIN_PATH,
                                          standardize_columns='numeric',
                                          ravel=False,
                                          selected_columns=None)
    X_test, y_test_, w_test = load_data(TEST_PATH,
                                       standardize_columns='numeric',
                                       ravel=False,
                                       selected_columns=None)

    index = np.concatenate((X_train.index, X_test.index))
    pov_rate.index = pd.Index(index)
#    pov_rate.to_csv('deep_fm_pov_rate.csv')

    ################################ once model is saved can probably drop this

    print(f"\n\n\t\tTEST SCORE f1: {f1(y_test, y_pred)}\n\n")
    evaluate_model(y_test, 
                   y_pred, 
                   y_prob, 
                   model_name=ALGORITHM_NAME, 
                   country=COUNTRY,
                   predict_pov_rate=False,
                   store_model=False)
    print("\n\n\n")


    # if we want to compare the other models

    # ------------------ FM Model ------------------
    #fm_params = dfm_params.copy()
    #fm_params["use_deep"] = False
    #y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


    # ------------------ DNN Model ------------------
    #dnn_params = dfm_params.copy()
    #dnn_params["use_fm"] = False
    #y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)
