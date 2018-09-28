
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# for reproducibility"
random_state = np.random.RandomState(2925)
np.random.seed(2925) 


# In[2]:

def make_country_df(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['mwi']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds, # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# # Models

# In[3]:

def model_mwi_v1(Xtr, Ytr, Xte):
   
    cat_list = list(Xtr.select_dtypes(include=['object', 'bool']).columns)

    le = LabelEncoder()

    for col in cat_list:
        le.fit(np.concatenate((Xtr[col].values, Xte[col].values), axis=0))
        Xtr[col] = pd.Categorical(le.transform(Xtr[col].values))
        Xte[col] = pd.Categorical(le.transform(Xte[col].values))

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(Xtr,
                      label=Ytr,
                     feature_name=list(Xtr.columns),
                      categorical_feature=cat_list) 
                                
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 43,
        'max_depth':16,
        'min_data_in_leaf': 16,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.56,
        'bagging_freq': 1,
        'lambda_l2':0.0, 
        'verbose' : 0,
        'seed':1,
        'learning_rate': 0.004,
        'num_threads': 24,
    }

    # train
    gbm = lgb.train(params, lgb_train, categorical_feature=cat_list, num_boost_round=3200)

    Yt = gbm.predict(Xte)
    return Yt


# # Data Processing

# In[4]:

data_paths = {
    'mwi': {
        'train_hhold': '../../data/raw_mwi/mwi_aligned_hhold_train.csv',
        'test_hhold':  '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
        'train_indiv': '../../data/raw_mwi/mwi_aligned_indiv_train.csv',
        'test_indiv':  '../../data/raw_mwi/mwi_aligned_indiv_test.csv'
    }
}

def get_cat_summary_choose(data_hhold, data_indiv, which='max', which_var=[], traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor', 'country']
    elif traintest=='test':
        var2drop = ['iid', 'country']
    varobj = which_var
    df = pd.DataFrame(index = data_hhold.index)
    for s in varobj:
        if which=='max':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').max()
        elif which=='min':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').min()
        else:
            print('Not a valid WHICH')
        # New formatting to support raw data
        df_s.columns = df_s.columns.map(lambda x: f'{s}__{x}')
        df = df.merge(df_s, left_index=True, right_index=True, suffixes=['', s+'_'])
    return df


# In[5]:

def get_features(Country='mwi', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    ## Add indiv features:
    if f_dict.get('cat_hot'):
        df = get_cat_summary_choose(data_hhold, data_indiv, which='min',
                             which_var = f_dict.get('cat_hot_which'),
                             traintest=traintest)
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
        df = get_cat_summary_choose(data_hhold, data_indiv, which='max',
                             which_var = f_dict.get('cat_hot_which'),
                             traintest=traintest)
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
    
    return data_hhold


# In[6]:

def pre_process_data(df, enforce_cols=None):
    
    df.drop(["country"], axis=1, inplace=True)
    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(df.mean(), inplace=True)
    
    return df


# In[7]:

def read_test_train_v2():

    feat = dict()
    feat['mwi'] = dict()
    feat['mwi']['cat_hot'] = True
    feat['mwi']['cat_hot_which'] = ['ind_rwenglish', 'ind_work6', 'ind_health5', 'ind_educ01', 'ind_work2']  # ['CaukPfUC', 'MUrHEJeh', 'MzEtIdUF', 'XizJGmbu', 'rQWIpTiG']
        
    mwi_train = get_features(Country='mwi', f_dict=feat['mwi'], traintest='train')  
    mwi_test = get_features(Country='mwi', f_dict=feat['mwi'], traintest='test')  
       
    print("Country A")
    mwiX_train = pre_process_data(mwi_train.drop('poor', axis=1))
    mwiy_train = np.ravel(mwi_train.poor)


    # process the test data
    mwiX_test = pre_process_data(mwi_test, enforce_cols=mwiX_train.columns)

#     aremove_list = ['sDGibZrP', 'RMbjnrlm', 'GUvFHPNA', 'iwkvfFnL', 'goxNwvnG', 'HDMHzGif', 'MOIscfCf',
#                     'tOWnWxYe', 'CtFxPQPT', 'fsXLyyco', 'ztGMreNV', 'YDgWYWcJ', 'pQmBvlkz', 'RLKqBexO', 
#                     'BwkgSxCk', 'rfDBJtIz', 'cOSBrarW', 'lRGpWehf', 'dSALvhyd', 'WbxAxHul', 'NitzgUzY', 
#                     'bhFgAObo', 'mnIQKNOM', 'GYTJWlaF', 'lTAXSTys', 'IBPMYJlv', 'WbEDLWBH', 'cgJgOfCA', 
#                     'hTraVEWP', 'nKoaotpH', 'OnTaJkLa', 'EMDSHIlJ', 'NGOnRdqc', 'vmZttwFZ', 'tjrOpVkX', 
#                     'zXPyHBkn', 'dkoIJCbY', 'hJrMTBVd', 'xNUUjCIL', 'rnJOTwVD', 'dAaIakDk', 'WqhniYIc', 
#                     'HfOrdgBo', 'wBXbHZmp', 'FGYOIJbC', 'CbzSWtkF', 'TzPdCEPV', 'lybuQXPm', 'GDUPaBQs',
#                     'EfkPrfXa', 'JeydMEpC', 'jxSUvflR', 'VFTkSOrq', 'CpqWSQcW', 'iVscWZyL', 'JMNvdasy', 
#                     'NrvxpdMQ', 'nGMEgWyl', 'pyBSpOoN', 'zvjiUrCR', 'aCfsveTu', 'TvShZEBA', 'TJUYOoXU', 
#                     'sYIButva', 'cWNZCMRB', 'yeHQSlwg', 'nSzbETYS', 'CVCsOVew', 'UXSJUVwD', 'FcekeISI', 
#                     'QBJeqwPF', 'mBlWbDmc', 'MBQcYnjc', 'KHzKOKPw', 'LrDrWRjC', 'TFrimNtw', 'InULRrrv', 
#                     'fhKiXuMY', 'fxbqfEWb', 'GnUDarun', 'XVwajTfe', 'yHbEDILT', 'JbjHTYUM', 'mHzqKSuN',
#                     'ncjMNgfp', 'dkPWxwSF', 'dsIjcEFe', 'ySkAFOzx', 'QzqNzAJE', 'bgfNZfcj', 'tZKoAqgl', 
#                     'NrUWfvEq', 'SsZAZLma', 'mNrEOmgq', 'hESBInAl', 'ofhkZaYa', 'mDTcQhdH', 'mvGdZZcs', 
#                     'ALbGNqKm', 'wgWdGBOp', 'nuwxPLMe', 'vRIvQXtC', 'rAkSnhJF', 'rtPrBBPl', 'tMJrvvut', 
#                     'BbKZUYsB', 'LjvKYNON', 'uZGqTQUP', 'NIRMacrk', 'UBanubTh', 'dEpQghsA', 'WiwmbjGW', 
#                     'ULMvnWcn', 'AsEmHUzj', 'BMmgMRvd', 'QqoiIXtI', 'duayPuvk', 'YKwvJgoP', 'ytYMzOlW',
#                     'YXkrVgqt', 'sslNoPlw', 'IIEHQNUc', 'ErggjCIN', 'tlxXCDiW', 'eeYoszDM', 'KAJOWiiw', 
#                     'UCnazcxd', 'uVnApIlJ', 'ZzUrQSMj', 'nGTepfos', 'ogHwwdzc', 'eoNxXdlZ', 'kZVpcgJL', 
#                     'lFcfBRGd', 'UXhTXbuS', 'UsENDgsH', 'wxDnGIwN', 'rYvVKPAF', 'OybQOufM', 'wnESwOiN', 
#                     'glEjrMIg', 'iBQXwnGC', 'VBjVVDwp', 'lOujHrCk', 'wakWLjkG', 'RJFKdmYJ', 'ZmJZXnoA', 
#                     'lQQeVmCa', 'ihGjxdDj', 'mycoyYwl', 'FlBqizNL', 'CIGUXrRQ', 'YlZCqMNw', 'gllMXToa',
#                     'DbUNVFwv', 'EuJrVjyG', 'uRFXnNKV', 'gfmfEyjQ', 'ggNglVqE']    

    # Refer to world-bank-ml-project/pover-t-tests/malawi/data/raw_mwi/Ordered%20Features%20Map.ipynb
    # for the mapping details
    mwi_remove_list = ['cons_0209', 'cons_1339', 'cons_0115', 'cons_0116', 'own_526', 'own_515', 'cons_0708', 'cons_0709',
                     'cons_0117', 'cons_1415', 'cons_1407', 'own_519', 'cons_0804', 'cons_0206', 'cons_0815',
                     'farm_611', 'cons_1337', 'cons_1412', 'cons_1338', 'cons_0309', 'cons_0706', 'cons_0823',
                     'hld_busin6', 'cons_0813', 'cons_1210', 'own_506', 'own_520', 'cons_1335', 'cons_0830',
                     'cons_1216', 'cons_0114', 'cons_0903', 'cons_0412', 'cons_0704', 'hld_mtltel', 'cons_0609',
                     'cons_0910', 'cons_1334', 'cons_0818', 'cons_0707', 'cons_0909', 'cons_0514', 'cons_1406',
                     'cons_1212', 'own_532', 'cons_0902', 'cons_0310', 'hld_busin8', 'cons_0110', 'own_505',
                     'cons_0208', 'own_530', 'own_531', 'own_514', 'cons_1218', 'farm_614', 'cons_0509', 'cons_0817',
                     'cons_0515', 'cons_1333', 'cons_0610', 'cons_0814', 'cons_0604', 'cons_0810', 'cons_1414',
                     'hld_credit2', 'inc_114', 'cons_0705', 'hld_whynoelec', 'farm_618', 'cons_0828', 'cons_1312',
                     'cons_0107', 'cons_0811', 'cons_1106', 'cons_1306', 'cons_1410', 'cons_1316', 'cons_1311',
                     'cons_1215', 'cons_1401', 'cons_0822', 'farm_622', 'cons_0825', 'own_524', 'hld_toilet',
                     'cons_0414', 'farm_610', 'own_521', 'cons_0103', 'cons_1209', 'inc_105', 'cons_0608', 'cons_1307',
                     'inc_111', 'cons_0906', 'hld_busin5', 'inc_104', 'cons_0820', 'own_511', 'cons_1315', 'hld_roof',
                     'com_distprimary', 'cons_1329', 'cons_0306', 'hld_busin2', 'hld_floor', 'farm_609', 'cons_1331',
                     'farm_601', 'cons_1416', 'inc_107', 'com_urbancenter', 'hld_headsleep', 'cons_0405', 'cons_1413',
                     'cons_0307', 'geo_urbrur', 'cons_1327', 'farm_625', 'cons_1301', 'cons_1330', 'cons_1319',
                     'own_523', 'cons_0409', 'inc_112', 'com_postoffice', 'cons_1313', 'cons_1221', 'cons_1309',
                     'cons_0407', 'cons_1317', 'cons_0406', 'own_522', 'gifts203', 'com_medicines', 'cons_1325',
                     'cons_1402', 'cons_0411', 'inc_103', 'cons_1102', 'cons_1105', 'farm_602', 'cons_1418', 'own_516',
                     'cons_1403', 'own_508', 'own_527', 'cons_0827', 'cons_1318', 'cons_0201', 'own_504', 'gifts202',
                     'farm_603', 'cons_1211', 'own_503', 'hld_adeqhous', 'hld_toiletshr', 'cons_0401']
    
    mwiX_train.drop(mwi_remove_list, axis=1, inplace=True)
    mwiX_test.drop(mwi_remove_list, axis=1, inplace=True)
    
    print("--------------------------------------------")
    return mwiX_train,mwiy_train, mwiX_test


# In[8]:

mwiX_train, mwiY_train, mwiX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[9]:

model = {'mwi':'model_mwi_v1'}

datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F08_'


# ## Submission

# In[10]:

mwi_preds = eval(model['mwi'])(mwiX_train, mwiY_train, mwiX_test)


# In[11]:

# convert preds to data frames
mwi_sub = make_country_df(mwi_preds.flatten(), mwiX_test, 'mwi')


# In[12]:

mwi_sub.to_csv(datafiles['out'] + '_mwi_test.csv')

