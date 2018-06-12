# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# import itertools
from IPython.display import display

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model as KerasLoadModel


from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score
)

from sklearn.metrics import (
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_curve,
    auc
)

from data.load_data import load_data, get_country_filepaths
from visualization.visualize import display_model_comparison

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           os.pardir, os.pardir))

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

def clip_yprob(y_prob):
    """Clip yprob to avoid 0 or 1 values. Fixes bug in log_loss calculation
    that results in returning nan."""
    eps = 1e-15
    y_prob = np.array([x if x <= 1-eps else 1-eps for x in y_prob])
    y_prob = np.array([x if x >= eps else eps for x in y_prob])
    return y_prob

def calculate_metrics(y_test, y_pred, y_prob=None, sample_weights=None):
    """Cacluate model performance metrics"""

    # Dictionary of metrics to calculate
    metrics = {}
    metrics['confusion_matrix']  = confusion_matrix(y_test, y_pred, sample_weight=sample_weights)
    metrics['roc_auc']           = None
    metrics['accuracy']          = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['precision']         = precision_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['recall']            = recall_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['f1']                = f1_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['cohen_kappa']       = cohen_kappa_score(y_test, y_pred)
    metrics['cross_entropy']     = None
    metrics['fpr']               = None
    metrics['tpr']               = None
    metrics['auc']               = None

    # Populate metrics that require y_prob
    if y_prob is not None:
        clip_yprob(y_prob)
        metrics['cross_entropy']     = log_loss(y_test,
                                                clip_yprob(y_prob), 
                                                sample_weight=sample_weights)
        metrics['roc_auc']           = roc_auc_score(y_test,
                                                     y_prob, 
                                                     sample_weight=sample_weights)

        fpr, tpr, _ = roc_curve(y_test,
                                y_prob, 
                                sample_weight=sample_weights)
        metrics['fpr']               = fpr
        metrics['tpr']               = tpr
        metrics['auc']               = auc(fpr, tpr, reorder=True)

    return metrics

def load_model_metrics(model_name, country):
    filepath = os.path.join(MODELS_DIR, country, model_name + '.pkl')
    with open(filepath, "rb") as f:
        m = pickle.load(f)
        m_metrics = calculate_metrics(m['y_true'],
                                      m['y_pred'],
                                      m['y_prob'],
                                      m['sample_weights'])
        m_metrics['name'] = m['name']
        m_metrics['pov_rate_error'] = m['pov_rate_error']
    return m_metrics

def load_model(model_name, country):
    filepath = os.path.join(MODELS_DIR, country, model_name + '.pkl')
    with open(filepath, "rb") as f:
        model = pickle.load(f)
        if type(model['model']) == str:
            model_path = os.path.join(MODELS_DIR, country, model_name + '.h5')
            model['model'] = KerasLoadModel(model_path)
    return model


def evaluate_model(y_test,
                   y_pred,
                   y_prob=None,
                   sample_weights=None,
                   show=True,
                   compare_models=None,
                   store_model=False,
                   model_name=None,
                   prefix=None,
                   country=None,
                   model=None,
                   features=None,
                   predict_pov_rate=True):
    """Evaluate model performance. Options to display results and store model"""

    metrics = calculate_metrics(y_test, y_pred, y_prob, sample_weights)

    # Predict national poverty rate
    pov_rate_pred = None
    if (predict_pov_rate == True) & (country is not None):
        selected_columns = features
        if type(features) in [pd.DataFrame, pd.Series]:
            selected_columns = features.index.values

        TRAIN_PATH, TEST_PATH, _ = get_country_filepaths(country)
        pov_rate_actual, pov_rate_pred = predict_poverty_rate(TRAIN_PATH,
                                                TEST_PATH,
                                                model,
                                                selected_columns=selected_columns,
                                                show=False,
                                                return_values=True)
        metrics['pov_rate_error'] = pov_rate_pred - pov_rate_actual
    else:
        metrics['pov_rate_error'] = None

    # Provide an output name if none given
    if model_name is None:
        model_name = 'score'
    if prefix is not None:
        model_name = prefix + "_" + model_name
    metrics['name'] = model_name

    # Display results
    if show is True:

        # Load models to compare
        comp_models = [metrics]
        if compare_models is not None:
            for comp_model in np.ravel(compare_models):
                filepath = os.path.join(MODELS_DIR, country, comp_model + '.pkl')
                with open(filepath, "rb") as f:
                    m = pickle.load(f)
                    m_metrics = calculate_metrics(m['y_true'],
                                                  m['y_pred'],
                                                  m['y_prob'],
                                                  m['sample_weights'])
                    m_metrics['name'] = m['name']
                    m_metrics['pov_rate_error'] = m['pov_rate_error']
                    comp_models.append(m_metrics)
        display_model_comparison(comp_models, show_roc=(y_prob is not None))

        if (predict_pov_rate == True) & (country is not None):
            print("Actual poverty rate: {:0.2%} ".format(pov_rate_actual))
            print("Predicted poverty rate: {:0.2%} ".format(pov_rate_pred))

    # Store model
    if (store_model is True) & (model_name is not None) & (country is not None):
        country_dir = os.path.join(MODELS_DIR, country)

        if not os.path.exists(country_dir):
            os.makedirs(country_dir)

        filepath = os.path.join(country_dir, model_name + '.pkl')
        with open(filepath, 'wb') as f:
            if type(model) == KerasClassifier:
                model_path = os.path.join(MODELS_DIR, country, model_name + '.h5')
                model.model.save(model_path)
                model = model_path
            if 'pov_rate_error' not in metrics.keys():
                metrics['pov_rate_error'] = None
            output = {'model': model,
                      'y_true': y_test,
                      'y_pred': y_pred,
                      'y_prob': y_prob,
                      'sample_weights': sample_weights,
                      'features': features,
                      'pov_rate_error': metrics['pov_rate_error'],
                      'timestamp': pd.Timestamp.utcnow(),
                      'name': model_name}
            pickle.dump(output, f)

    return metrics

def compare_algorithm_models(algorithm_name, country, include_simple=False):
    files = os.listdir(os.path.join(MODELS_DIR, country))
    files = [f[:-4] for f in files if f[-4:] == '.pkl']
    if include_simple == False:
        files = [f for f in files if f[-6:] != 'simple']
    files = [f for f in files if f[0:len(algorithm_name)] == algorithm_name]

    metrics = [load_model_metrics(f, country) for f in files]
    results = display_model_comparison(metrics,
                                       show_roc=True,
                                       show_cm=False,
                                       show_pov_rate_error=True,
                                       transpose=True)


def predict_poverty_rate(train_path, test_path, model,
                         standardize_columns='numeric',
                         ravel=True,
                         selected_columns=None,
                         show=True,
                         return_values=False):
    # Recombine the entire dataset to get the actual poverty rate
    X_train, y_train, w_train = load_data(train_path,
                                          standardize_columns=standardize_columns,
                                          ravel=ravel,
                                          selected_columns=selected_columns)
    X_test, y_test, w_test = load_data(test_path,
                                       standardize_columns=standardize_columns,
                                       ravel=ravel,
                                       selected_columns=selected_columns)
    pov_rate = pd.DataFrame(np.vstack((np.vstack((y_train, w_train)).T,
                                       np.vstack((y_test, w_test)).T)),
                            columns=['poor', 'wta_pop'])
    pov_rate_actual = (pov_rate.wta_pop * pov_rate.poor).sum() / pov_rate.wta_pop.sum()

    # Make predictions on entire dataset to get the predicted poverty rate
    pov_rate['pred'] = model.predict(np.concatenate((X_train.as_matrix(), X_test.as_matrix())))
    pov_rate_pred = (pov_rate.wta_pop * pov_rate.pred).sum() / pov_rate.wta_pop.sum()

    if show == True:
        print("Actual poverty rate: {:0.2%} ".format(pov_rate_actual))
        print("Predicted poverty rate: {:0.2%} ".format(pov_rate_pred))
    if return_values:
        return pov_rate_actual, pov_rate_pred
    else:
        return

def load_feats(models, country):
    feats = {}
    n_feats = 0
    for f in models:
        model = load_model(f, country)
        if (model['features'] is not None) and (len(model['features']) > n_feats):
            n_feats = len(model['features'])
        feats[model['name']] = model['features']

    return feats
