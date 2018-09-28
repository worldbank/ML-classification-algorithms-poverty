import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import class_weight
from abc import ABC, abstractmethod


class predict_model(ABC):
    """
    Abstract class for working with classifiers.
    """

    @abstractmethod
    def __init__(self, name='predict_model', categ_conv=True):
        self.params = {}
        self.exclude_list = []
        self.name = name
        self.random = 1
        self.classifier = None
        self.categ_conv = categ_conv
        self.data_df = {}

    def set_params(self, params=None):
        if not params:
            self.params = {}
        else:
            self.params = params

    def set_random_seed(self, random=1):
        self.random = random

    @abstractmethod
    def load_data(self, data, balance=False):
        self.data = data

        self.data_df['train'], self.data_df['y'] = self.data.get_train(
                                                        balance=balance
                                                   )
        self.data_df['test'] = self.data.get_test()

        self.category_cols = self.data.get_cat_list()
        for header in self.category_cols:
            self.data_df['train'].loc[:, header] = self.data_df['train'][header].astype('category').cat.codes
            self.data_df['test'].loc[:, header] = self.data_df['test'][header].astype('category').cat.codes
        return True

    def get_train(self):
        return self.data_df['train']

    def get_y(self):
        return self.data_df['y']

    def get_test(self):
        return self.data_df['test']

    def set_exclude_list(self, exclude_list):
        self.exclude_list = exclude_list.copy()

    @abstractmethod
    def get_feature_importances(self):
        pass

    @abstractmethod
    def train(self, x_train=None, y_train=None):
        pass

    def predict(self, test=None):
        if self.classifier:
            if not isinstance(test, pd.DataFrame):
                test = self.get_test()
            elif self.categ_conv:
                cols = [x for x in self.category_cols if x in test.columns]
                for header in cols:
                    test.loc[:, header] = test[header].astype('category').cat.codes
            test = test.drop(
                [x for x in self.exclude_list if x in test.columns], axis=1
            )
            res = pd.DataFrame(index=test.index)
            res['country'] = self.data.country
            res['poor'] = self.classifier.predict_proba(test)[:, 1]
            return res
        else:
            print('error: classifier not defined')
            return None


class CB_model(predict_model):
    """
        Class for a CatBoost classifier.
    """

    def __init__(self, name='cat_boost', categ_conv=True):
        super().__init__(name='cat_boost', categ_conv=categ_conv)
        self.name = name

    def load_data(self, data, balance=False):
        if super().load_data(data, balance):
            c_w = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(self.data_df['y']),
                y=self.data_df['y']
            )

            self.classifier = CatBoostClassifier(**self.params,
                                                 class_weights=c_w)
            return True
        else:
            return False

    def train(self, x_train=None, y_train=None):

        if not isinstance(x_train, pd.DataFrame):
            x_train = self.get_train()
        elif self.categ_conv:
            cols = [x for x in self.category_cols if x in x_train.columns]
            for header in cols:
                x_train.loc[:, header] = x_train[header].astype('category').cat.codes

        if not isinstance(y_train, pd.Series):
            y_train = self.get_y()

        x_train = x_train.drop([x for x in self.exclude_list
                                if x in x_train.columns], axis=1)

        self.category_cols = [x for x in self.category_cols
                              if x not in self.exclude_list]

        cat_dims = [x_train.columns.get_loc(i) for i in self.category_cols]
        print(x_train.shape, y_train.shape, len(self.category_cols))

        self.classifier.fit(x_train, y_train, cat_features=cat_dims)
        return self.classifier

    def get_feature_importances(self):
        return self.classifier._feature_importance


class XGB_model(predict_model):
    """
        Class for a XGBoost classifier.
    """

    def __init__(self, name='xg_boost', categ_conv=True):
        super().__init__(name='xg_boost', categ_conv=categ_conv)
        self.name = name

    def load_data(self, data, balance=False):
        if super().load_data(data, balance):
            self.params['scale_pos_weight'] = (
                (self.data_df['y'].shape[0] - self.data_df['y'].sum()) /
                self.data_df['y'].sum()
            )
            self.classifier = xgb.XGBClassifier(**self.params)
            return True
        else:
            return False

    def train(self, x_train=None, y_train=None):

        if not isinstance(x_train, pd.DataFrame):
            x_train = self.get_train()
        elif self.categ_conv:
            cols = [x for x in self.category_cols if x in x_train.columns]
            for header in cols:
                x_train.loc[:, header] = x_train[header].astype('category').cat.codes

        if not isinstance(y_train, pd.Series):
            y_train = self.get_y()

        x_train = x_train.drop([x for x in self.exclude_list
                                if x in x_train.columns], axis=1)
        print('x_train shape: ', x_train.shape)

        self.classifier.fit(x_train, y_train)

        return self.classifier

    def get_feature_importances(self):
        return self.classifier.feature_importances_


class LGBM_model(predict_model):
    """
        Class for LightGBM classifier.
    """

    def __init__(self, name='lgbm', categ_conv=True):
        super().__init__(name='lgbm', categ_conv=categ_conv)
        self.name = name

    def load_data(self, data, balance=False):
        if super().load_data(data, balance):
            self.classifier = lgb.LGBMClassifier(**self.params)
            return True
        else:
            return False

    def train(self, x_train=None, y_train=None):

        if not isinstance(x_train, pd.DataFrame):
            x_train = self.get_train()
        elif self.categ_conv:
            cols = [x for x in self.category_cols if x in x_train.columns]
            for header in cols:
                x_train.loc[:, header] = x_train[header].astype('category').cat.codes

        if not isinstance(y_train, pd.Series):
            y_train = self.get_y()

        x_train = x_train.drop([x for x in self.exclude_list
                                if x in x_train.columns], axis=1)
        print('x_train shape: ', x_train.shape)

        self.category_cols = [x for x in self.category_cols
                              if x not in self.exclude_list]

        self.classifier.fit(x_train, y_train, verbose=False)

        return self.classifier

    def get_feature_importances(self):
        return self.classifier.feature_importances_
