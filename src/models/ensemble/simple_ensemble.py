from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(Path(__file__).parent.parent)
import models.evaluation as evaluation

COUNTRY = 'idn'
LIST_OF_MODELS = ['lr_full_oversample',
 'mlp_full_undersample_cv',
 'lr_full_oversample_cv',
 'xgb_full_undersample_cv',
 'lr_full_undersample',
 'lr_full_classwts',
 'mlp_full_undersample',
 'lda_full_oversample_cv',
 'lr_l1_feats_oversample_cv',
 'lda_full_oversample']

def simple_ensemble_preds(X_test, list_of_models=LIST_OF_MODELS):
    
    # load the data
    country_dir = Path(evaluation.MODELS_DIR, COUNTRY)
    models = [evaluation.load_model(str(f.stem), COUNTRY)
              for f in country_dir.iterdir() if f.stem in list_of_models]

    all_preds_test = pd.DataFrame(np.zeros((X_test.shape[0], len(list_of_models))),
                             index=X_test.index,
                             columns=list_of_models).astype(float)
    
    # make preds on train and test
    for model in models:
        if 'feat' in model['name']:
            reduced_feat = model['features'].index.values
            # X_train_reduced = X_train[reduced_feat]
            X_test_reduced = X_test[reduced_feat]
            # all_preds_train[model['name']] = model['model'].predict_proba(X_train_reduced.as_matrix())[:, 1]
            all_preds_test[model['name']] = model['model'].predict_proba(X_test_reduced.as_matrix())[:, 1]
        else:
            # all_preds_train[model['name']] = model['model'].predict_proba(X_train.as_matrix())[:, 1]
            all_preds_test[model['name']] = model['model'].predict_proba(X_test.as_matrix())[:, 1]

    # make poverty rate prediction
    # means_train = all_preds_train.mean(axis=1)
    means_test = all_preds_test.mean(axis=1)
    
    # ensemble_train = np.round(means_train)
    ensemble_test = np.round(means_test)

    simple_ensemble_prob = all_preds_test.mean(axis=1)
    return simple_ensemble_prob.as_matrix()