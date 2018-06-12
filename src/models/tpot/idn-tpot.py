import json
import logging
import os
import sys
import time

import click
import numpy as np
import pandas as pd
from tpot import TPOTClassifier

# python idn-tpot.py --max_time_mins=3600  --pop_size=100
# python idn-tpot.py --max_time_mins=1  --pop_size=20
# python idn-tpot.py --sample=True --sample_size=5000 --gens=5 --pop_size=100

sys.path.append("../../")
from data.load_data import (get_country_filepaths,
                            split_features_labels_weights,
                            load_data)
from features.process_features import get_vif, standardize

COUNTRY = 'idn'
TRAIN_PATH, TEST_PATH, QUESTIONS_PATH = get_country_filepaths(COUNTRY)


@click.command()
@click.option('--gens', default=100, help='set number of generations')
@click.option('--pop_size', default=100, help='set size of population')
@click.option('--max_time_mins', default=None, help='how many minutes TPOT has to optimize the pipeline, will override gen if not None')
@click.option('--sample', default=False, help="test on subset of training")
@click.option('--sample_size', default=1000, help="choose sample size")
@click.option('--export_best_pipeline', default=True, help="export result")
def run_tpot_idn_optimization(gens,
                              pop_size,
                              max_time_mins,
                              sample,
                              sample_size,
                              export_best_pipeline=True):


    # Initiate logging - overwrite existing log file if exists
    logging.basicConfig(filename="tpot_idn_pipeline.log", 
                        level=logging.INFO, 
                        filemode='w')

    # Load and transform the training data
    X_train, y_train, w_train = load_data(TRAIN_PATH)

    # Load and transform the test data
    X_test, y_test, w_test = load_data(TEST_PATH)

    logging.info("Train and test data loaded")

    orig_index = X_train.index
    # Reduce training set if testing sample
    if sample:
        orig_train_size = X_train.shape[0]
        X_train = (X_train.reset_index()
                          .sample(sample_size, random_state=0)
                          .set_index('hid'))

        logging.info(f"Training samples reduced from {orig_train_size} to {X_train.shape[0]}")

    # Assert consistency before training
    assert X_train.shape[1] == X_test.shape[1]

    # Enforce consistency between features and labels
    y_train = pd.DataFrame(y_train, index=orig_index).loc[X_train.index]
    y_train = np.ravel(y_train.values)

    if max_time_mins is not None:
        max_time_mins = int(max_time_mins)

    # Instantiate tpot classifier
    logging.info("Instantiating classifier")
    tpot = TPOTClassifier(generations=gens,
                          population_size=pop_size,
                          max_time_mins=max_time_mins,
                          scoring='f1',
                          verbosity=2,
                          n_jobs=-1,
#                          memory='auto',
                          random_state=2017,
#                          early_stop=6,
                          subsample=0.5)

    # Fit the classifier
    start_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
    logging.info(f"Fitting classifier:\t\t{start_time}")
    
    tpot.fit(X_train, y_train, w_train)
    
    end_time = time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime())
    logging.info(f"Fitting complete:\t\t{end_time}")
    
    # Export the pipeline
    logging.info("Exporting best pipeline...")
    if export_best_pipeline:
        tpot.export('tpot_idn_pipeline.py')

    # Record f1 score on test set
    logging.info(f"Test f1 score:\t{tpot.score(X_test, y_test)}")


if __name__ == '__main__':
    run_tpot_idn_optimization()
