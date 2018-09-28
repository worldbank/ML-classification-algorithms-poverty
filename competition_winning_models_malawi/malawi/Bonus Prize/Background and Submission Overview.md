![Banner Image](https://s3.amazonaws.com/drivendata/comp_images/wb-poverty.png)
# Pover-T Tests: Predicting Poverty - World Bank
<br> <br>


# Entrant Background and Submission Overview

### Who are you (mini-bio) and what do you do professionally?
I'm a Data Scientist currently specializing in AI&mdash;primarily in the field of NLP and Information Retrieval. I currently work as a Senior Manager for R&D and leading an R&D team working on applying AI in job matching and recruitment at [Kalibrr](https://www.kalibrr.com). I have a Masters degree in Physics and worked on time-series and complex systems before. I enjoy binging on blogs and youtube videos about AI and its applications to NLP problems.

### High level summary of your approach: what did you do and why?
This competition is very challenging due to the very anonymous nature of the data. Only very limited feature engineering can be done. Also, without any comprehensible insights about the features, I just decided to use an L1 regularized Logistic Regression model (with its sparse solutions in mind) to set a benchmark. It ended up working pretty well with the country A dataset.

I also threw in the usual models such as Random Forest, XGBoost, LightGBM, and Neural Networks. The tree based models worked pretty well in country B and C datasets&mdash;most likely due to the heavily imbalanced nature of the two datasets.

At first, my plan is to perform Bayesian optimization on the parameters to come up with a single best parameter set for each. Later, I realized that I can just combine the best performing variation of the Bayesian optimized models. I ended up building 100 variations by taking the top 20 variations of each model as the meta-models to be used in the final predictions.

The common method in combining these many models is stacking. However, stacking surprisingly didn't work well. It took me a while to realize that I can fairly easily combine an arbitrary number of model results by blending them. Common appoach in blending is by manually assigning weights based on intuition. While this method works for fairly small number of models, this is not scalable to quite a number of models&mdash;in this case 100. I was able to solve this problem via a constrained optimization of the model weights.

In the weight optimization process, I used the out-of-fold predictions from each meta-models and constrained the weights such that they sum up to one. The coefficients of the linear combination of the out-of-fold predictions are then optimized based on the loss metric against the actual values. This process was done using stratified cross validation (10-folds) and the coefficients for each fold were then averaged to blend the test meta-predictions.

I think the model could have scored higher if I used more meta-models instead of just selecting the best 20 for each base model. :D


### Copy and paste the 3 most impactful parts of your code and explain what each does and how it helped your model.

**Bayesian optimization function**: this code helped me to scale the creation and the selection of meta-models for each base model by performing Bayesian optimization over the defined parameter space.

```python
def bayesian_optimize_model(country_code, model_type, tunable_params=None, num_iter=50, init_points=0):
    global round_num
    global is_training

    round_num = 0
    is_training = True

    store_fname = 'xgbBO_{}_res_{}_optimization_{}.dict'.format(country_code, model_type, datetime.now())
    logging.info('Bayesian optimization results will be stored in {} after training...'.format(store_fname))

    if tunable_params is None:
        if model_type == 'xgb':
            tunable_params = {
                'indiv_corr_thresh': (0, 0.3),
                'colsample_bytree': (0.2, 1),
                'max_depth': (2, 6),
                'subsample': (0.2, 1),
                'gamma': (0, 2),
                'scale_pos_weight': (0, 1),
            }
        elif model_type == 'lr':
            tunable_params = {
                'indiv_corr_thresh': (-0.9, 0.3),
                'C': (0.001, 0.6),
            }
        elif model_type == 'lgb':
            tunable_params = {
                'indiv_corr_thresh': (0, 0.4),
                'num_leaves': (4, 64),
                'colsample_bytree' : (0.2, 1),
                'subsample' : (0.2, 1),
                'min_child_samples': (2, 120),
                'scale_pos_weight': (0, 1),
            }
        elif model_type == 'nn':
            tunable_params = {
                'indiv_corr_thresh': (-0.9, 0.3),
                'l1_num': (0, 100),
                'l2_num' : (0, 100),
                'l3_num' : (0, 100),
                'alpha' : (0.005, 0.1),
            }

        elif model_type == 'rf':
            tunable_params = {
                'indiv_corr_thresh': (0, 0.4),
                'max_depth': (2, 20),
                'min_samples_split' : (2, 20),
                'min_samples_leaf' : (2, 20),
            }

    if model_type == 'xgb':
        model_predict = xgb_predict
    elif model_type == 'lr':
        model_predict = lr_predict
    elif model_type == 'lgb':
        model_predict = lgb_predict
    elif model_type == 'nn':
        model_predict = nn_predict
    elif model_type == 'rf':
        model_predict = rf_predict

    modelBO = BayesianOptimization(
        model_predict, tunable_params, verbose=bayes_opt_verbose
    )

    modelBO.maximize(init_points=init_points, n_iter=num_iter, acq="poi", xi=0.1)

    with open('./bayesian-opts-res/{:0.5}'.format(-modelBO.res['max']['max_val']) + '-' + store_fname, 'w') as fl:
        cPickle.dump(modelBO.res, fl)

    return modelBO
```


**Model blending optimizer**: this code allowed me to combine the predictions from the meta-models in a more principled way making blending of models scalable. I also consider this as one of my personal innovations that came up from this competition.

```python
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
```

**Model parallelization**: some of the models I used do not have a native parallelization support. As such, I implemented the code below to speed-up the training and prediction of the models.

```python
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
```


### What are some other things you tried that didn’t necessarily make it into the final workflow (quick overview)?

I tried using likelihood encoding for the categorical features but it didn't make the models perform better.

Also, I initially tried using Keras as the framework for the Neural Net but I ended up using the sklearn implementation. Sklearn's implementation is quite promising!


### Did you use any tools for data preparation or exploratory data analysis that aren’t listed in your code submission?
No, I decided to work with a model that involves very minimal external input. Feature selection were done statistically without visual plotting, e.g., cut-off of aggregated `indiv` data was done based on correlation as well as using Logistic Regression.


### How did you evaluate performance of the model other than the provided metric, if at all?
Imperatively, all of my model evaluation is centered around the `log loss` metric. The Bayesian optimization process optimizes the log loss of an optimization fold. The blending weights to combine the out-of-fold predictions were also optimized using the `log loss` metric.

I've been comparing the distribution of test predictions per country but I wouldn't really count that as a model evaluation since I have no way of measuring whether the model actually improves or not. :D


### Potentially Helpful Features
I think giving some insights about what the features were would have been very helpful. In any case, I designed the model itself to be agnostic to the features and simply used statistics and allow the models to learn the relevant ones. I think a more involved feature selection could have improved my model's performance!


### Anything we should watch out for or be aware of in using your model (e.g. code quirks, memory requirements, numerical stability issues, etc.)?
The model training will run for quite some time as it performs the Bayesian optimization over the base models, probably about 10 hours. Also, a lot of random stuff is happening in the model but I tried my best to control it with seeds. Try to train in a machine with at least 8 cpu core / threads and at least 16GB of ram just to be on the safe side. :)

The functions defined in `Full Bayesian Model Training and Predictions.ipynb` notebook rely on global variables. As such, be careful in refactoring!


### Do you have any useful charts, graphs, or visualizations from the process?
None


### If you were to continue working on this problem for the next year, what methods or techniques might you try in order to build on your work so far? Are there other fields or features you felt would have been very helpful to have?
I will definitely put more effort in the feature selection. Also, if the features were to be unobfuscated, feature engineering would also be part of improving the model.

In the model aspect, I think increasing the parameter space and allowing the Bayesian optimizer to perform more exploration (instead of exploitation) will be a good allocation of time. Doing so will most likely allow solutions to reach other local optima of the models&mdash;hopefully adding to the overall robustness of the predictions.

Adding in some models might be useful. Example, I would incorporate ElasticNet in the blend to optimize the `L1` and `L2` regularization jointly instead of just using LASSO Logistic Regression model as in the current solution.
