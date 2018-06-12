import multiprocessing
multiprocessing.set_start_method('forkserver')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Binarizer, Normalizer
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier

import sys
sys.path.append("../../")
from data.load_data import (get_country_filepaths,
                            split_features_labels_weights,
                            load_data)
from features.process_features import get_vif, standardize

sys.path.append("../")
from evaluation import calculate_metrics, evaluate_model

COUNTRY = 'idn'
TRAIN_PATH, TEST_PATH, QUESTIONS_PATH = get_country_filepaths(COUNTRY)

# Load and transform the training data
X_train, y_train, w_train = load_data(TRAIN_PATH)

# Load and transform the test data
X_test, y_test, w_test = load_data(TEST_PATH)
orig_index = X_train.index


# Assert consistency before training
assert X_train.shape[1] == X_test.shape[1]

# Enforce consistency between features and labels
y_train = pd.DataFrame(y_train, index=orig_index).loc[X_train.index]
y_train = np.ravel(y_train.values)



# NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1).values
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'].values, random_state=42)

training_features = X_train
testing_features = X_test
training_target = y_train
testing_target = y_test

# Score on the training set was:0.6090057503598979
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LogisticRegression(C=5.0, dual=False, penalty="l2")),
    StackingEstimator(estimator=XGBClassifier(learning_rate=1.0, max_depth=1, min_child_weight=12, n_estimators=100, nthread=1, subsample=0.7500000000000001)),
    Normalizer(norm="max"),
    StackingEstimator(estimator=LogisticRegression(C=0.5, dual=False, penalty="l1")),
    StackingEstimator(estimator=LogisticRegression(C=0.5, dual=False, penalty="l1")),
    Binarizer(threshold=0.25),
    XGBClassifier(learning_rate=0.001, max_depth=1, min_child_weight=10, n_estimators=100, nthread=1, subsample=1.0)
)

exported_pipeline.fit(training_features, training_target)

y_pred = exported_pipeline.predict(testing_features).astype(np.float64)
y_prob = exported_pipeline.predict_proba(testing_features)[:, 1].astype(np.float64)


evaluate_model(y_test.astype(np.float64), 
               y_pred, 
               y_prob, 
               model_name='tpot', 
               country=COUNTRY,
               predict_pov_rate=True,
               store_model=True,
               show=False,
               model=exported_pipeline)
print("done")