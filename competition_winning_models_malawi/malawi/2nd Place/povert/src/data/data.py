import pandas as pd
from sklearn.utils import resample
from collections import OrderedDict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


class Data():
    """ Class for working with households data."""

    def __init__(self):
        self.country_df_train = None
        self.country_df_test = None
        self.categorical_list = []
        self.float_list = []
        self.train_file_name = None
        self.test_file_name = None

    def split_data(self,
                   size=0.8,
                   n_splits=1,
                   random_state=1,
                   balance=False,
                   df=None):
        """
        Returns data partitions.

        Args:
            size: float, partition ratio, optional (default=0.8)
            n_splits: int, number of partitions, optional (default=1)
            random_state: int, RandomState instance, optional (default=1)
            balance: bool, resample data, optional (default=False)
            df: DataFrame, data for split, optional (default=None)

        Returns:
            List of splits.
        """

        if not isinstance(df, pd.DataFrame):
            train = self.country_df_train
        else:
            train = df
        sss = StratifiedShuffleSplit(n_splits=n_splits,
                                     test_size=1-size,
                                     random_state=random_state)
        splits = []
        for train_index, validate_index in sss.split(train, train.poor):
            df_train = train.iloc[train_index]
            if balance:
                df_train = self.resample(df_train)
            splits.append((df_train, train.iloc[validate_index]))
        return splits

    def _rename_col(self):
        """Rename columns."""

        train_columns = self.country_df_train.columns
        train_new_columns = [
            x if (x == 'poor' or
                  x == 'country') else '{0}_{1}'.format(
                      self.country,
                      train_columns.get_loc(x)) for x in train_columns]
        self.country_df_train.columns = train_new_columns
        self.col_maping = dict(zip(train_columns, train_new_columns))
        self.col_maping_reverse = dict(zip(train_new_columns, train_columns))

        self.country_df_test.rename(columns=self.col_maping, inplace=True)

    def del_nonunique(self, df):
        """
        Delete columns with non-unique values.

        Args:
            df: DataFrame, data for clean

        Returns:
            DataFrame without columns with non-unique values.
        """

        nunique = df.apply(pd.Series.nunique)
        cols_to_drop = nunique[nunique == 1].index
        print('Cols to drop:', cols_to_drop)
        return df.drop(cols_to_drop, axis=1)

    def category_float_search(self,
                              count=5,
                              countries=['mwi'],
                              cat_types=['object'],
                              fi_types=['float64', 'int64']):
        """
        Search for categorical features.

        Args:
            count: int, number of unique values for determining categoricity,
                   optional (default=5)
            countries: list, list of countries for which to search not only for
                       features with the type in cat_types,
                       optional (default=['mwi'])
            cat_types: list, list of types for categorical features,
                       optional (default=['object'])
            fi_types:  list, A list of additional types for searching for category
                       features, optional (default=['float64', 'int64'])

            Returns:
                Tuple with a list of categorical columns and a list of other columns
        """

        categorical_list = list(
            self.country_df_train[self.col_common_list].select_dtypes(
                include=cat_types).columns)

        if self.country not in countries:
            return (categorical_list,
                    list(self.country_df_train[
                        self.col_common_list].select_dtypes(
                            include=fi_types).columns))

        float_list = []
        print('float list length: ', len(list(
            self.country_df_test.select_dtypes(include=fi_types).columns)))
        for i in list(self.country_df_test[
                self.col_common_list].select_dtypes(include=fi_types).columns):
            value_set = set(
                self.country_df_test[i].unique()).union(set(
                    self.country_df_train[i].unique()))
            if len(value_set) <= count:
                categorical_list.append(i)
            else:
                float_list.append(i)
        print('float list length: ', len(sorted(float_list)))
        return sorted(categorical_list), sorted(float_list)

    def scale(self):
        """
            Scale all non categorical values.
        """
        if not self.float_list:
            print('There is no float list')
            return False
        scaler = StandardScaler()
        for i in self.float_list:
            self.country_df_train[i] = scaler.fit_transform(
                self.country_df_train[i].values.reshape(-1, 1))
            self.country_df_test[i] = scaler.transform(
                self.country_df_test[i].values.reshape(-1, 1))
        return True

    def fillna(self):
        """
            Replace `NaN` values with the median of the column and remove all the completely empty columns.
        """
        print('train data have NaNs: ', self.country_df_train.isnull().any().any())
        print('test data have NaNs: ', self.country_df_test.isnull().any().any())
        self.country_df_train = self.country_df_train.fillna(
            self.country_df_train.median()).dropna(axis=1, how='all')
        self.country_df_test = self.country_df_test.fillna(
            self.country_df_test.median()).dropna(axis=1, how='all')
        print('train data have NaNs: ', self.country_df_train.isnull().any().any())
        print('test data have NaNs: ', self.country_df_test.isnull().any().any())

    def set_file_names(self, files_dict):
        """
        Set file names for train and test dataframes

        Args:
            files_dict: dictionary, file names for 'train' and 'test'
        """
        self.train_file_name = files_dict.get('train')
        self.test_file_name = files_dict.get('test')

    def set_country(self, country):
        """
        Set country label.

        Args:
            country: string, a label for country
        """
        self.country = country
        print('Country: ', self.country)

    def load(self, load=True, with_bug=True):
        """
        Load data from files.

        Args:
            load: bool, load from file without postprocessing,
                  optional (default=True)
            with_bug: bool, emulate a bug for final submission,
                  optional, (default=True)
        """
        self.country_df_train = self.del_nonunique(
            pd.read_csv(self.train_file_name, index_col='id'))
        self.country_df_test = self.del_nonunique(
            pd.read_csv(self.test_file_name, index_col='id'))

        if not load:
            self._rename_col()
            self.fillna()
        self.col_common_list = \
            sorted(list(set(self.country_df_train.columns).intersection(
                self.country_df_test.columns)))
        self.categorical_list, self.float_list = self.category_float_search()
        if not load:
            if not with_bug:
                self.scale()
        print('dataind train shape: ', self.country_df_train.shape)
        return True

    def save(self, files_dict, poor=True):
        """
        Save data to files.

        Args:
            files_dict: dictionary, file names for 'train' and 'test'
            poor: bool, save poor column, optional (default=True)
        """
        train = self.get_train()
        if poor:
            train = pd.concat([train[0], train[1]], axis=1)
        else:
            train = train[0]
        train.to_csv(files_dict.get('train'), index=True, mode='w')
        test = self.get_test()
        test.to_csv(files_dict.get('test'), index=True, mode='w')
        return True

    def resample(self, df):
        """
        Resample dataframe.

        Args:
            df: DataFrame, dataframe for resample

        Returns:
            Resampled dataframe.
        """
        df_majority = df[~self.country_df_train.poor]
        df_minority = df[self.country_df_train.poor]

        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=df_majority.shape[0],
                                         random_state=1)
        return pd.concat([df_majority, df_minority_upsampled])

    def get_train(self, balance=False):
        """
        Get train data.

        Args:
            balance: bool, resample data, optional (default=False)

        Returns:
            Tuple with a train dataframe and a target dataframe.
        """
        if balance:
            train = self.resample(self.country_df_train)
            return train[self.col_common_list], train['poor']
        return (self.country_df_train[self.col_common_list],
                self.country_df_train['poor'])

    def get_train_valid(self, n_splits=1, balance=False):
        """
        Get train and valid sets.

        Args:
            n_splits: int, number of partitions, optional (default=1)
            balance: bool, resample data, optional (default=False)

        Returns:
            A list of splits.
        """
        splits = self.split_data(n_splits=n_splits, balance=balance)
        return [((x[self.col_common_list], x.poor),
                 (y[self.col_common_list], y.poor)) for x, y in splits]

    def get_test(self):
        """
        Get test data.

        Returns:
            A test dataframe.
        """
        return self.country_df_test[self.col_common_list]

    def get_cat_list(self):
        """
        Get a list of categorical features.

        Returns:
            A list of columns.
        """
        return self.categorical_list

    def get_float_list(self):
        """
        Get a list of non-categorical features.

        Returns:
            A list of columns.
        """
        return self.float_list


class DataInd(Data):
    """ Class for working with individual level data."""

    def __init__(self):
        super().__init__()

    def get_poor(self, df):
        """
        Get a dataframe with poor column.

        Returns:
            A dataframe with a poor column.
        """
        return df['poor'].reset_index()[['id', 'poor']].drop_duplicates().set_index('id')

    def summarize(self, df):
        """
        Get a dataframe with a summarized individual level data for household.

        Args:
            df: DataFrame, dataframe with an individual level data

        Returns:
            A dataframe with summarized columns.
        """
        count = df.copy().groupby(level=0).sum()
        res_df = pd.concat({'sum': count}, axis=1)
        res_df.columns = ['{0}_{1}'.format(i[0], i[1]) for i in res_df.columns]
        res_df = res_df.reindex(index=df.index.get_level_values(0))
        res_df = res_df[~res_df.index.duplicated(keep='first')]
        print('summarized size df: ', res_df.shape)
        return res_df

    def _get_id_list(self, df):
        """
        Get an ordered list of indeces.

        Args:
            df: DataFrame, dataframe with an individual level data

        Returns:
            An ordered list of indeces.
        """
        return list(OrderedDict.fromkeys(df.index.get_level_values(0)))

    def count_iid(self, df):
        """
        Get a dataframe with a count of individuals for households.

        Args:
            df: DataFrame, dataframe with an individual level data

        Returns:
            A dataframe with a count of individuals for households.
        """
        s = df.index.get_level_values(0).value_counts()
        return s.reindex(index=self._get_id_list(df)).to_frame('iid_cnt')

    def count_neg_poz(self, df):
        """
        Get a dataframe with a count of negative and positive values for
        an individual level data.

        Args:
            df: DataFrame, dataframe with an individual level data

        Returns:
            A dataframe with a count of negative and positive values for
            an individual level data.
        """
        res_df = df.select_dtypes(include=['float64', 'int64', 'int8'])
        res_df = res_df.groupby(level=0).apply(lambda c: c.apply(
                lambda x: pd.Series(
                    [(x < 0).sum(), (x >= 0).sum()])).unstack())
        res_df.columns = ['{0}_{1}'.format(i[0], i[1])
                          for i in res_df.columns]
        print('count_neg_poz size df: ', res_df.shape)
        return res_df.reindex(index=self._get_id_list(df))

    def count_unique_categories(self, df, iid=True):
        """
        Get a dataframe with a count of unique values for an individual
        level data.

        Args:
            df: DataFrame, dataframe with an individual level data
            iid: bool, add columns with the ratio of the number of unique
                 values to the number of individuals in households,
                 optional (default=True)

        Returns:
            A dataframe with a count of unique values for an individual
            level data.
        """
        res_df = df.groupby(level=0).apply(
            lambda c: c.apply(lambda x: pd.Series([len((x).unique())])))
        res_df.index = res_df.index.droplevel(1)
        res_df.columns = [
            '{0}_{1}'.format('cat_n', i) for i in res_df.columns]
        print('count_unique_categories size df: ', res_df.shape)
        res_df = res_df.reindex(index=self._get_id_list(df))
        if iid:
            div_df = res_df.div(self.count_iid(df)['iid_cnt'], axis=0)
            div_df.columns = ['{0}_{1}'.format('div_cat_iid', i)
                              for i in res_df.columns]
            res_df = pd.concat([res_df, div_df], axis=1)
        return res_df

    def load(self, load=True, cat_enc=False):
        """
        Load data from files.

        Args:
            load: bool, load from file without postprocessing,
                  optional (default=True)
            cat_enc: bool, encode categories to numeric values,
                  optional, (default=False)
        """

        print('DataInd load')
        if load:
            self.country_df_train = self.del_nonunique(
                pd.read_csv(self.train_file_name, index_col=['id']))
            self.country_df_test = self.del_nonunique(
                pd.read_csv(self.test_file_name, index_col=['id']))

        if not load:
            print(self.train_file_name)
            print(self.test_file_name)
            self.country_df_train = self.del_nonunique(
                pd.read_csv(self.train_file_name, index_col=['id', 'iid']))
            self.country_df_test = self.del_nonunique(
                pd.read_csv(self.test_file_name, index_col=['id', 'iid']))
            self._rename_col()
            self.fillna()
            self.col_common_list = sorted(
                list(set(self.country_df_train.columns).intersection(
                        self.country_df_test.columns)))

            self.categorical_list, self.float_list = self.category_float_search(
                countries=['mwi'])

            if cat_enc:
                for header in self.categorical_list:
                    self.country_df_train[header] = self.country_df_train[header].astype('category').cat.codes
                    self.country_df_test[header] = self.country_df_test[header].astype('category').cat.codes
            # To reproduce the result in the final submission.
            # In the general solution, this scale is not needed.
            self.scale()
            self.country_df_train = self.del_nonunique(pd.concat(
                [self.get_poor(self.country_df_train),
                 self.count_iid(self.country_df_train),
                 self.count_neg_poz(self.country_df_train),
                 self.summarize(self.country_df_train),
                 self.count_unique_categories(self.country_df_train)],
                axis=1))

            self.country_df_test = self.del_nonunique(pd.concat(
                [self.count_iid(self.country_df_test),
                 self.count_neg_poz(self.country_df_test),
                 self.summarize(self.country_df_test),
                 self.count_unique_categories(self.country_df_test)],
                axis=1))

        self.col_common_list = sorted(
            list(set(self.country_df_train.columns).intersection(
                self.country_df_test.columns)))
        self.categorical_list, self.float_list = self.category_float_search(
            countries=['mwi'])
        if not load:
            self.scale()
        print('indiv train shape: ', self.country_df_train.shape)
        print('indiv test shape: ', self.country_df_test.shape)
        return True


class DataConcat(Data):
    """
    Class for working with concatenated data from individual and household
    levels.
    """

    def __init__(self):
        self.data_hh_train = None
        self.data_hh_test = None
        self.data_indiv_train = None
        self.data_indiv_test = None
        super().__init__()

    def set_file_names(self, files_dict):
        """
        Set file names for train and test dataframes

        Args:
            files_dict: dictionary, file names for 'train' and 'test'
        """
        self.hh_train_file_name = files_dict.get('train_hh')
        self.hh_test_file_name = files_dict.get('test_hh')
        self.ind_train_file_name = files_dict.get('train_ind')
        self.ind_test_file_name = files_dict.get('test_ind')
        super().set_file_names(files_dict)

    def load(self, load=True, cat_enc=False, with_bug=True):
        """
        Load data from files.

        Args:
            load: bool, load from file without postprocessing,
                  optional (default=True)
            cat_enc: bool, encode categories to numeric values,
                  optional, (default=False)
            with_bug: bool, emulate a bug for final submission,
                  optional, (default=True)
        """
        if with_bug or not load:
            data_hh = Data()
            data_hh.set_country(self.country)
            data_hh.set_file_names({'train': self.hh_train_file_name,
                                    'test': self.hh_test_file_name})
            if not data_hh.load(load=False, with_bug=with_bug):
                return False

        if load:
            print('DataConcat load')
            self.country_df_train = self.del_nonunique(pd.read_csv(
                self.train_file_name, index_col=['id']))
            self.country_df_test = self.del_nonunique(pd.read_csv(
                self.test_file_name, index_col=['id']))
        else:
            data_ind = DataInd()
            data_ind.set_country(self.country)
            data_ind.set_file_names({'train': self.ind_train_file_name,
                                     'test': self.ind_test_file_name})

            if data_ind.load(load=True):
                self.country_df_train = data_hh.country_df_train.join(
                    data_ind.country_df_train)
                self.country_df_test = data_hh.country_df_test.join(
                    data_ind.country_df_test)

        self.col_common_list = sorted(
            list(set(self.country_df_train.columns).intersection(
                self.country_df_test.columns)))

        if with_bug:
            self.categorical_list = data_hh.categorical_list
        else:
            self.categorical_list, self.float_list = self.category_float_search(
                                                         countries=['mwi'])

        print('train:', self.country_df_train.shape)
        print('test:', self.country_df_test.shape)

        return True
