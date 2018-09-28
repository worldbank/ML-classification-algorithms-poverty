
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats

random_state = np.random.RandomState(2925)
np.random.seed(2925) # for reproducibility"

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import  LabelEncoder

from keras.regularizers import l2,l1
from keras.layers import Input, Embedding, Dense, Dropout, Flatten
from keras.models import Model
from keras.layers.core import Lambda
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.layers.advanced_activations import PReLU


# In[2]:

def make_country_df(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# Next feature transformations were inspired by this post: [link](https://medium.com/airbnb-engineering/overcoming-missing-values-in-a-random-forest-classifier-7b1fc1fc03ba)

# In[3]:

def make_Fx_numeric(x_train, x_test, y_train):
    x_train_nonmissing_loc = (x_train != -1212121)
    x_test_nonmissing_loc = (x_test != -1212121)
    x_train_nonmissing = x_train[x_train_nonmissing_loc] 
    x_test_nonmissing = x_test[x_test_nonmissing_loc]
    
    y_train_nonmissing = y_train[x_train_nonmissing_loc]
    y_train_nonmissing_ones = (y_train_nonmissing == True)
    x_train_nonmissing_ones = x_train_nonmissing[y_train_nonmissing_ones]
    
    prob_train = [round(stats.percentileofscore(x_train_nonmissing_ones, a, 'weak')/100.,2) for a in x_train_nonmissing]
    prob_test = [round(stats.percentileofscore(x_train_nonmissing_ones, a, 'weak')/100.,2) for a in x_test_nonmissing]
    
    Fx_train = np.array(x_train[:], dtype=float)
    Fx_test = np.array(x_test[:], dtype=float)
    Fx_train[x_train_nonmissing_loc]=prob_train
    Fx_test[x_test_nonmissing_loc]=prob_test
    return Fx_train, Fx_test


# # Models

# In[4]:

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])


# In[5]:

# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    return df


def keras_encoding(df_train,df_test):

    ntrain = df_train.shape[0]
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    
    num_list = list(df_train.select_dtypes(include=['int64', 'float64']).columns)
    df_all = standardize(df_all)
    
    cat_list = list(df_train.select_dtypes(include=['object', 'bool']).columns)
    for c in cat_list:
        df_all[c] = df_all[c].astype('category').cat.as_ordered()
        
    le = LabelEncoder()

    for col in cat_list:
        le.fit(df_all[col].values)
        df_all[col] = le.transform(df_all[col].values)

    Din = dict()
    Dout = dict()   
    for col in cat_list:
        cat_sz = np.size(np.unique(df_all[col].values))
        Din[col]= cat_sz
        Dout[col] = max(3,min(50, (cat_sz+1)//2))
    
    df_train = df_all.iloc[:ntrain,:].copy()
    df_test = df_all.iloc[ntrain:,:].copy()
    return df_train,df_test, num_list, cat_list, Din, Dout


# In[6]:

def Keras_A01(Xtr,Ytr,Xte,num_list, cat_list, Din, Dout,cv_i):

    X_list = []
    for col in cat_list:
        X_list.append(Xtr[col].values)
    X_list.append(Xtr[num_list].values)
    X_train = X_list
    X_list = []
    for col in cat_list:
        X_list.append(Xte[col].values)
    X_list.append(Xte[num_list].values)
    X_test = X_list
    l2_emb = 0.0001

    #emb_layers=[]
    cat_out = []
    cat_in = []

    #cat var
    for idx, var_name in enumerate(cat_list):
        x_in = Input(shape=(1,), dtype='int64', name=var_name+'_in')

        input_dim = Din[var_name]
        output_dim = Dout[var_name]
        x_out = Embedding(input_dim, 
                          output_dim, 
                          input_length=1, 
                          name = var_name, 
                          embeddings_regularizer=l1(l2_emb))(x_in)

        flatten_c = Flatten()(x_out)
        #emb_layers.append(x_out) 
        
        cat_in.append(x_in)
        cat_out.append(flatten_c)  

    x_emb = layers.concatenate(cat_out,name = 'emb')

    #continuous variables
    cont_in = Input(shape=(len(num_list),), name='continuous_input')
    cont_out = Lambda(expand_dims, expand_dims_output_shape)(cont_in)
    x_num = Flatten(name = 'num')(cont_out)

    cat_in.append(cont_in)

    #merge
    x = layers.concatenate([x_emb,x_num],name = 'emb_num')
    x = Dense(512)(x)
    x = PReLU()(x)
    x = Dropout(0.6)(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(32)(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)


    model = Model(inputs = cat_in, outputs = x)
    
    model.compile(optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, Ytr, batch_size=256, epochs=9,verbose=0,shuffle=True)
 
    Yt = model.predict(X_test).flatten() 
    K.clear_session()
    return Yt


# In[7]:

def batch_generator(X, y,cat_list,num_list,batch_size):
    
    n_splits = X.shape[0] // (batch_size - 1)

    skf = StratifiedKFold(n_splits=n_splits,random_state=2925, shuffle=True)

    while True:
        for idx_tr, idx_te in skf.split(X, y):
            X_batch = X.iloc[idx_te].reset_index(drop=True).copy()
            y_batch = y[idx_te]
        
            X_list = []
            for col in cat_list:
                X_list.append(X_batch[col].values)
            X_list.append(X_batch[num_list].values)
            X_batch = X_list    

            yield (X_batch, y_batch)


def Keras_C01(Xtr,Ytr,Xte,num_list, cat_list, Din, Dout,cv_i):
    X_list = []
    for col in cat_list:
        X_list.append(Xtr[col].values)
    X_list.append(Xtr[num_list].values)
    X_train = X_list
    
    X_list = []
    for col in cat_list:
        X_list.append(Xte[col].values)
    X_list.append(Xte[num_list].values)
    X_test = X_list

    l2_emb = 0.0001

    #emb_layers=[]
    cat_out = []
    cat_in = []

    #cat var
    for idx, var_name in enumerate(cat_list):
        x_in = Input(shape=(1,), dtype='int64', name=var_name+'_in')

        input_dim = Din[var_name]
        output_dim = Dout[var_name]
        x_out = Embedding(input_dim, 
                          output_dim, 
                          input_length=1, 
                          name = var_name, 
                          embeddings_regularizer=l1(l2_emb))(x_in)

        flatten_c = Flatten()(x_out)
        #emb_layers.append(x_out) 
        
        cat_in.append(x_in)
        cat_out.append(flatten_c)  
        
    x_emb = layers.concatenate(cat_out,name = 'emb')

    #continuous variables
    cont_in = Input(shape=(len(num_list),), name='continuous_input')
    cont_out = Lambda(expand_dims, expand_dims_output_shape)(cont_in)
    x_num = Flatten(name = 'num')(cont_out)

    cat_in.append(cont_in)

    #merge
    x = layers.concatenate([x_emb,x_num],name = 'emb_num')
    x = Dense(512)(x)
    x = PReLU()(x)
    x = Dropout(0.6)(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    x = Dropout(0.3)(x)
    x = Dense(32)(x)
    x = PReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = cat_in, outputs = x)
    
    model.compile(optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    batch_size = 256
    model.fit_generator(generator=batch_generator(Xtr, Ytr,cat_list,num_list, batch_size),
                    epochs=9, verbose=0, steps_per_epoch= np.floor(Xtr.shape[0]/batch_size))

    Yt = model.predict(X_test).flatten()
    K.clear_session()
    return Yt


# # Data Processing

# In[8]:

def pre_process_data(df, enforce_cols=None):
    #print("Input shape:\t{}".format(df.shape))
    df.drop(["country"], axis=1, inplace=True)

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})
    
    df.fillna(df.mean(), inplace=True)
    
    return df


# In[9]:

data_paths = {'A': {'train_hhold': 'data/A_hhold_train.csv', 
                        'test_hhold':  'data/A_hhold_test.csv',
                        'train_indiv': 'data/A_indiv_train.csv', 
                        'test_indiv':  'data/A_indiv_test.csv'}, 

                  'B': {'train_hhold': 'data/B_hhold_train.csv', 
                        'test_hhold':  'data/B_hhold_test.csv',
                        'train_indiv': 'data/B_indiv_train.csv', 
                        'test_indiv':  'data/B_indiv_test.csv'}, 

                  'C': {'train_hhold': 'data/C_hhold_train.csv', 
                        'test_hhold':  'data/C_hhold_test.csv',
                        'train_indiv': 'data/C_indiv_train.csv', 
                        'test_indiv':  'data/C_indiv_test.csv'}}


def get_hhold_size(data_indiv):
    return data_indiv.groupby('id').country.agg({'hhold_size':'count'})


def get_num_mean(data_indiv, traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor']
    elif traintest=='test':
        var2drop = ['iid']
    return data_indiv.drop(var2drop, axis=1).groupby('id').mean()

def get_num_summary(data_indiv, which=None, traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor']
    elif traintest=='test':
        var2drop = ['iid']
 
    aux = ~data_indiv.drop(var2drop, axis=1).dtypes.isin(['object','bool', 'O'])
    varnum = [aux.keys()[i] for i,j in enumerate(aux) if aux.values[i]]

    data_num_max = data_indiv[varnum].groupby('id').max()
    
    if which=='max':
        ans = data_indiv[varnum].groupby('id').max()
    elif  which=='min':
        ans = data_indiv[varnum].groupby('id').min()
    return ans


def get_cat_summary_choose(data_hhold, data_indiv, which='max', which_var=[], traintest=None):
    var2drop = []
    if traintest=='train':
        var2drop = ['iid', 'poor', 'country']
    elif traintest=='test':
        var2drop = ['iid', 'country']

    varobj = which_var
    df = pd.DataFrame(index = data_hhold.index)
    for s in varobj:
        #print(s)
        if which=='max':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').max()
        elif which=='min':
            df_s = pd.get_dummies(data_indiv[s]).groupby('id').min()
        else:
            print('Not a valid WHICH')
        #print(df_s.keys())
        df = df.merge(df_s, left_index=True, right_index=True, suffixes=['', s+'_'])
    return df


# In[10]:


def get_features(Country='A', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    varobj = data_indiv.select_dtypes('object', 'bool').columns

    ## Add indiv features:
    #hhold size
    if f_dict.get('hh_size'):
        data_hh_size = get_hhold_size(data_indiv)
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
    ## mean of numerical
    if f_dict.get('num_mean'):
        data_num_mean = get_num_mean(data_indiv, traintest=traintest)
        data_hhold = data_hhold.merge(data_num_mean, left_index=True, right_index=True)
   
    # max of numerical
    if f_dict.get('num_max'):
        data_num_max = get_num_summary(data_indiv, which='max', traintest=traintest)
        data_hhold = data_hhold.merge(data_num_max, left_index=True, right_index=True, suffixes=['', '_max'])
   
    # min of numerical
    if f_dict.get('num_min'):
        data_num_min = get_num_summary(data_indiv, which='min', traintest=traintest)
        data_hhold = data_hhold.merge(data_num_min, left_index=True, right_index=True, suffixes=['', '_min'])
       
    if f_dict.get('cat_hot'):
        df = get_cat_summary_choose(data_hhold, data_indiv, which='min',
                             which_var = varobj,
                             traintest=traintest)
        df = df.add_suffix('_ind')
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)

        df = get_cat_summary_choose(data_hhold, data_indiv, which='max',
                             which_var = varobj,
                             traintest=traintest)
        df = df.add_suffix('_ind')
        data_hhold = data_hhold.merge(df, left_index=True, right_index=True)
        
    return data_hhold


# In[11]:

def read_test_train_v2():

    feat = dict()
    feat['A'] = dict()
    feat['A']['hh_size'] = True
    feat['A']['num_mean'] = True
    feat['A']['num_max'] = True
    feat['A']['num_min'] = True
    feat['A']['cat_hot'] = True
    feat['A']['cat_hot_which'] =  []
    
    a_train = get_features(Country='A', f_dict=feat['A'], traintest='train')  
    a_test = get_features(Country='A', f_dict=feat['A'], traintest='test')  
   
    
    #feat = dict()
    feat['C'] = dict()
    feat['C']['hh_size'] = True
    feat['C']['num_mean'] = True
    feat['C']['num_max'] = True
    feat['C']['num_min'] = True
    feat['C']['cat_hot'] = True
    feat['C']['cat_hot_which'] = []

    c_train = get_features(Country='C', f_dict=feat['C'], traintest='train')  
    c_test = get_features(Country='C', f_dict=feat['C'], traintest='test')  

    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor).astype(np.int8)

    print("\nCountry C")
    cX_train = pre_process_data(c_train.drop('poor', axis=1))
    cy_train = np.ravel(c_train.poor).astype(np.int8)

    # process the test data
    aX_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
    cX_test = pre_process_data(c_test, enforce_cols=cX_train.columns)
    
    Afeatures = ['SlDKnCuu','maLAYXwi', 'vwpsXRGk', 'TYhoEiNm', 'zFkComtB', 'zzwlWZZC', 'DxLvCGgv', 
                 'CbABToOI', 'uSKnVaKV', 'nzTeWUeM', 'nEsgxvAq', 'NmAVTtfA', 'YTdCRVJt', 'QyBloWXZ', 
                 'HKMQJANN', 'ZRrposmO', 'HfKRIwMb', 'UCAmikjV', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 
                 'ltcNxFzI', 'MxOgekdE', 'JwtIxvKg', 'bEPKkJXP', 'cqUmYeAp', 'sFWbFEso', 'TqrXZaOw', 
                 'galsfNtg', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT', 'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 
                 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'OMtioXZZ', 'bIBQTaHw', 'KcArMKAe', 'wwfmpuWA', 
                 'znHDEHZP', 'kWFVfHWP', 'XwVALSPR', 'HHAeIHna', 'dCGNTMiG', 'ngwuvaCV', 'XSgHIFXD', 
                 'ANBCxZzU', 'NanLCXEI', 'ZnBLVaqz', 'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'pWyRKfsb',
                 'udzhtHIr', 'IZFarbPw', 'lnfulcWk', 'QNLOXNwj', 'YFMZwKrU', 'RJQbcmKy', 'uizuNzbk', 
                 'dlyiMEQt', 'TnWhKowI', 'LoYIbglA', 'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 'gfurxECf', 
                 'hnrnuMte', 'LrQXqVUj', 'XDDOZFWf', 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 
                 'xkUFKUoW', 'jVDpuAmP', 'SeZULMCT', 'AtGRGAYi', 'WTFJilSZ', 'NBfffJUe', 'mvgxfsRb', 
                 'UXfyiodk', 'EftwspgZ', 'szowPwNq', 'BfGjiYom', 'iWEFJYkR', 'BCehjxAl', 'nqndbwXP', 
                 'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'tHFrzjai', 'MKozKLvT', 'pjHvJhoZ', 
                 'zkbPtFyO', 'xZBEXWPR', 'dyGFeFAg', 'pKPTBZZq', 'bCYWWTxH', 'EQKKRGkR', 'cCsFudxF', 
                 'muIetHMK', 'ishdUooQ', 'ItpCDLDM', 'ptEAnCSs', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts',
                 'JzhdOhzb', 'THDtJuYh', 'jwEuQQve', 'rQAsGegu', 'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 
                 'TiwRslOh', 'PWShFLnY', 'lFExzVaF', 'IKqsuNvV', 'CqqwKRSn', 'YUExUvhq', 'yaHLJxDD', 
                 'qlZMvcWc', 'dqRtXzav', 'ktBqxSwa', 'GIMIxlmv', 'wKVwRQIp', 'UaXLYMMh', 'bKtkhUWD', 
                 'HhKXJWno', 'tAYCAXge', 'aWlBVrkK', 'cDkXTaWP', 'hnmsRSvN', 'GHmAeUhZ', 'BIofZdtd', 
                 'QZiSWCCB', 'CsGvKKBJ', 'OLpGAaEu', 'JCDeZBXq', 'HGPWuGlV', 'WuwrCsIY', 'AlDbXTlZ', 
                 'hhold_size', 'ukWqmeSS', 'ukWqmeSS_max', 'ukWqmeSS_min', 'mOlYV_ind_x', 'msICg_ind_x', 
                 'YXCNt_ind_x', 'HgfUG_ind_x', 'EaHvf_ind_x', 'pdgUV_ind_x', 'xrEKh_ind_x', 'QkRds_ind_x', 
                 'XNPgB_ind_x', 'vvXmD_ind_x', 'KOjYm_ind_x', 'Qydia_ind_x', 'vtkRP_ind_x', 'RPBUw_ind_x', 
                 'QQdHS_ind_x', 'Hikoa_ind_x', 'SlRmt_ind_y', 'TRFeI_ind_y', 'fmdsF_ind_y', 'lBMrM_ind_y', 
                 'tMiQp_ind_y', 'wWIzo_ind_y', 'xnnDH_ind_y', 'CXizI_ind_y', 'DQhEE_ind_y', 'LvUxT_ind_y', 
                 'SSvEP_ind_y', 'YsahA_ind_y', 'lzzev_ind_y', 'ccbZA_ind_y', 'fOUHD_ind_y', 'vkRKJ_ind_y', 
                 'rwCRh_ind_y', 'yomtK_ind_y', 'iWGMu_ind_y', 'EaHvf_ind_y', 'GmSKW_ind_y', 'tymHY_ind_y', 
                 'yhUHu_ind_y', 'pdgUV_ind_y', 'qIbMY_ind_y', 'sDvAm_ind_y', 'bszTA_ind_y', 'veBMo_ind_y', 
                 'SowpV_ind_y', 'OeQKE_ind_y', 'XNPgB_ind_y', 'MxNAc_ind_y', 'SuzRU_ind_y', 'PmhpI_ind_y', 
                 'SjaWF_ind_y', 'TUafC_ind_y', 'bazjA_ind_y', 'dpMMl_ind_y', 'qVwNL_ind_y', 'zTqjB_ind_y', 
                 'BNylo_ind_y', 'CXjLj_ind_y', 'PwkMV_ind_y', 'Qydia_ind_y', 'kVYrO_ind_y', 'VneGw_ind_y', 
                 'rXEFU_ind_y', 'aKoLM_ind_y', 'SWhXf_ind_y', 'UCsCT_ind_y', 'uJdwX_ind_y', 'qmOVd_ind_y', 
                 'yOwsR_ind_y', 'ZIrnY_ind_y', 'dAmhs_ind_y', 'gCSRj_ind_y', 'ESfgE_ind_y', 'okwnE_ind_y', 
                 'OkXob_ind_y', 'dDnIb_ind_y', 'jVHyH_ind_y', 'xUYIC_ind_y']
    
    
    Cfeatures = ['vmKoAlVH', 'LhUIIEHQ', 'KIUzCiTC', 'NONtAKOM',  'zyABXhnz', 'gUzYYakV', 'FlsGEbwx', 
                 'WdGBOpdZ', 'kLAQgdly', 'TusIlNXO', 'tPfjVDpu', 'EQtGHLFz', 'gLDyDXsb', 'xFKmUXhu', 
                 'oniXaptE', 'QUnDQaUl', 'ubefFytO', 'zLrQXqVU', 'coFdvtHB', 'yBSpOoNe', 'wpgRhUno', 
                 'XYfcLBql', 'pQGrypBw', 'DBjxSUvf', 'avZMPHry', 'HDZTYhoE', 'wcNjwEuQ', 'phbxKGlB', 
                 'HNRJQbcm', 'GJGfnAWg', 'tIeYAnmc', 'LwKcZtLN', 'nRXRObKS', 'DMslsIBE', 'AJHrHUkH', 
                 'ihACfisf', 'obIQUcpS', 'mmoCpqWS', 'XKQWlRjk_max', 'vWNISgEA_max', 'XKQWlRjk_min', 
                 'bsMfXBld_min', 'xqUoo_ind_x', 'amOeQ_ind_x', 'RxYsa_ind_x', 'ucHNS_ind_x', 'GHDuu_ind_x', 
                 'dxzZA_ind_x', 'DGWjH_ind_x', 'XvXON_ind_x', 'LuEXv_ind_x', 'hRHpW_ind_x', 'kvMGu_ind_x', 
                 'rAwcc_ind_x', 'vtkRP_ind_x', 'xgpHA_ind_x', 'xRxWC_ind_x', 'BZKME_ind_y', 'uSuzR_ind_y', 
                 'izIlz_ind_y', 'lJvCX_ind_y', 'bCSuY_ind_y', 'ALcKg_ind_y', 'FoQcU_ind_y', 'GpnOQ_ind_y', 
                 'vhhVz_ind_y', 'EGPlQ_ind_y', 'EhzOz_ind_y', 'MyWVa_ind_y', 'UrHEJ_ind_y', 'ehUOC_ind_y', 
                 'gRXcL_ind_y', 'JnveI_ind_y', 'KEvSa_ind_y', 'hAGot_ind_y', 'Iwnmb_ind_y', 'tPmhp_ind_y', 
                 'ucqiX_ind_y', 'mlNXN_ind_y', 'niWGM_ind_y', 'qQkRd_ind_y', 'sMBUT_ind_y', 'yWhop_ind_y', 
                 'JskzT_ind_y', 'cPXrX_ind_y', 'yFSGe_ind_y', 'wsHHy_ind_y', 'hOlGY_ind_y', 'bgZsP_ind_y', 
                 'xyraV_ind_y', 'EPnnG_ind_y', 'pClPr_ind_y', 'FeIwW_ind_y', 'Izoay_ind_y', 'gvqxs_ind_y', 
                 'MZyJF_ind_y', 'QrjGn_ind_y', 'iuiyo_ind_y', 'NBQEn_ind_y', 'Ysraf_ind_y', 'fZCQS_ind_y', 
                 'sitaC_ind_y', 'wZsYz_ind_y', 'QGHnL_ind_y', 'xgpHA_ind_y', 'kXobL_ind_y', 'oacjJ_ind_y', 
                 'xRxWC_ind_y']    
   

    aX_train = aX_train[Afeatures].copy()
    aX_test = aX_test[Afeatures].copy()

    cX_train = cX_train[Cfeatures].copy()
    cX_test = cX_test[Cfeatures].copy()
    print("--------------------------------------------")
    return aX_train, ay_train, aX_test, cX_train, cy_train, cX_test


# In[12]:

aX_train, aY_train, aX_test, cX_train, cY_train, cX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[13]:

model = {'A':'Keras_A01','B':'Keras_B01','C':'Keras_C01'}

datafiles = {}
datafiles['out'] = 'predictions/Keras_M03_F09_'


# ## Submission

# In[14]:

aX_train,aX_test,anum_list, acat_list, aDin, aDout = keras_encoding(aX_train,aX_test)

#----------------------------------------------------------------
varofint = cX_train.select_dtypes(['int64','float64']).keys()
for v in varofint: 
    GX_train, GX_test =  make_Fx_numeric(cX_train[v].copy().values, cX_test[v].copy().values, cY_train)
    cX_train[v+'_Fx'] = GX_train
    cX_test[v+'_Fx'] = GX_test
#----------------------------------------------------------------
for v in varofint:
    aux = pd.concat([cX_train[v],cX_test[v]], axis=0)
    v2o = dict((j,i) for i,j in enumerate(np.sort(aux.unique())))
    cX_train[v+'_Ox'] = cX_train[v].apply(lambda s: v2o[s]) 
    cX_test[v+'_Ox'] = cX_test[v].apply(lambda s: v2o[s])
#----------------------------------------------------------------

cX_train,cX_test,cnum_list, ccat_list, cDin, cDout = keras_encoding(cX_train,cX_test)


# In[15]:

a_preds = eval(model['A'])(aX_train, aY_train, aX_test,anum_list, acat_list, aDin, aDout,0)
c_preds = eval(model['C'])(cX_train, cY_train, cX_test,cnum_list, ccat_list, cDin, cDout,0)


# In[16]:

# convert preds to data frames
a_sub = make_country_df(a_preds.flatten(), aX_test, 'A')
c_sub = make_country_df(c_preds.flatten(), cX_test, 'C')


# In[17]:

a_sub.to_csv(datafiles['out']+'_A_test.csv')
c_sub.to_csv(datafiles['out']+'_C_test.csv')


# In[ ]:




