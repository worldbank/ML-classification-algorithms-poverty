# Overview and Instructions for Result Replication


## Project Tree

```
world-bank-pover-t-tests-solution
├── Background and Submission Overview.md
├── data
│   └── get_data.sh
├── README.md
├── src
│   ├── bayesian-opts-res
│   │   └── bayesian-opt-test-preds
│   ├── Data Processor Original Dataset.ipynb
│   ├── Full Bayesian Model Training and Predictions.ipynb
│   └── modules
│       ├── __init__.py
│       ├── training_models.py
│       ├── training_optimizers.py
│       └── training_utils.py
└── submission

6 directories, 9 files
```


## Data


### Raw dataset
Place the training data inside the `data/` directory of the project. This can also be done automatically (assuming you're in the root directory) by running:

```
$ cd data/
$ bash get_data.sh
```

The data below should be present inside the `data/` directory in order to proceed to the next step of generating the transformed dataset for training.
```
│   ├── A_hhold_test.csv
│   ├── A_hhold_train.csv
│   ├── A_indiv_test.csv
│   ├── A_indiv_train.csv
│   ├── B_hhold_test.csv
│   ├── B_hhold_train.csv
│   ├── B_indiv_test.csv
│   ├── B_indiv_train.csv
│   ├── C_hhold_test.csv
│   ├── C_hhold_train.csv
│   ├── C_indiv_test.csv
│   ├── C_indiv_train.csv
```


### Transforming raw data for training

Assuming you're in the root directory, navigate inside the `src/` directory and open the `Data Processor Original Dataset.ipynb` notebook. The notebook will do the following transformations to the `hhold` and `indiv` datasets for each country.


**Process to generate *indiv_cat*:**

1. Take only categorical features
2. One-hot-encode the features
3. Summarize the encoded features to represent a household using:
    - `mean`
    - `median`
    - `all`
    - `any`

**Process to generate *hhold-transformed*:**

1. Take numeric and categorical data
2. For numeric, transform data using:
    - MinMax scaler: `mx_`
    - Standard scaler: `sc_`
3. For categorical, encode data:
    - Use label encoding
    - Use the label encoded data to perform one-hot-encoding
    - Retain the label encoding


The above process will generate these additional files inside the `data/` directory. These will be used by the models.

```
│   ├── A-hhold-transformed-test.csv
│   ├── A-hhold-transformed-train.csv
│   ├── B-hhold-transformed-test.csv
│   ├── B-hhold-transformed-train.csv
│   ├── C-hhold-transformed-test.csv
│   ├── C-hhold-transformed-train.csv
│   ├── indiv_cat_train.hdf
│   ├── indiv_cat_test.hdf
```



## Model

For each country, the model is a blending of meta predictions from 20 variations of 5 models.
The following base models are used:

* [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) with L1 regularization
* [Neural Network](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) (3 hidden layers)
* [Random Forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/model.html)

Each variation is produced by performing Bayesian optimization over the base models given a range of parameter values. The Bayesian optimization is trained to optimize the prediction score over an optimization fold. The optimization fold is allowed to randomly vary for a more robust model mixture to prevent overfitting which is likely to happen if only a single optimization fold is used.

The top 20 meta-models having the highest optimization-fold score are included in the blending model. The blending model is trained by optimizing the log loss of the out-of-fold (OOF) predictions against the actual values. The variables over which the optimization is made are the weights of each meta-model to the final prediction.


## Dependencies

* python version 2.7.12

This project depends on the following python modules:

* **Standard**:
    * os
    * datetime
    * glob
    * cPickle
    * time
    * warnings
    * hashlib
    * contextlib


* **Packages**:
    * numpy==1.14.0
    * pandas==0.20.2
    * joblib==0.11
    * bayesian-optimization==0.6.0
    * scikit-learn==0.19.0
    * xgboost==0.7
    * lightgbm==2.1.0
    * scipy==1.0.0
    * matplotlib==2.0.0
    * tqdm==4.11.2


## Replicating Results


Assuming you're in the root directory, navigate inside the `src/` directory and open the `Full Bayesian Model Training and Predictions.ipynb` notebook. Run all cells. This will take a while to complete.

The submission file will be generated and stored in the `submission/` directory in the project root.

Logs from the model training can be accessed by looking at the `output.logs` file.

## Other details

Please check the [Background and Submission Overview](./Background%20and%20Submission%20Overview.md) for more details.
