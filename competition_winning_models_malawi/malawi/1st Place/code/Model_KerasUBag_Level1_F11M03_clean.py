
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from scipy import stats

random_state = np.random.RandomState(2925)
np.random.seed(2925) # for reproducibility"

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

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
    country_codes = ['mwi']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds,  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


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

def Keras_mwi01(Xtr, Ytr, Xte):
    
    Xtr,Xte,num_list, cat_list, Din, Dout = keras_encoding(Xtr,Xte)
    
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

def batch_generator(X, y, cat_list, num_list, batch_size):
    
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

# In[8]:

def Bagging_Test(Xtr, Ytr, Xte, c):
    Yt_av =  np.zeros(Xte.shape[0])
    nbags = 3
    nfolds = 8
    kf = 0
    for i in range(nfolds):
        xtr,ytr = resample(Xtr,Ytr,n_samples=int(0.95 *Xtr.shape[0]),replace=False,random_state=10*i)
        pred = np.zeros(Xte.shape[0])
        for j in range(nbags):
            res = eval(models[c])(xtr,ytr,Xte).flatten()
            pred += res
            Yt_av += res
        pred /= nbags
        kf+=1
    Yt_av /= (nfolds*nbags)
    return Yt_av


# # Data Processing

# In[9]:

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


# In[10]:

data_paths = {
    'mwi': {
        'train_hhold': '../../data/raw_mwi/mwi_aligned_hhold_train.csv', 
        'test_hhold':  '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
        'train_indiv': '../../data/raw_mwi/mwi_aligned_indiv_train.csv', 
        'test_indiv':  '../../data/raw_mwi/mwi_aligned_indiv_test.csv'
    }             
}


# In[11]:

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
 
    aux = ~data_indiv.drop(var2drop, axis=1).dtypes.isin(['object', 'bool', 'O'])
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
    #print(var2drop)
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
        # New formatting to support raw data
        df_s.columns = df_s.columns.map(lambda x: f'{s}__{x}')
        df = df.merge(df_s, left_index=True, right_index=True, suffixes=['', s+'_'])
    return df


# In[12]:

def get_features(Country='mwi', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    varobj = data_indiv.select_dtypes('object', 'bool').columns

    ## Add indiv features:
    if f_dict.get('div_by_hh_size'):
        varofint = data_hhold.select_dtypes(['int', 'float']).keys()
        data_hh_size = get_hhold_size(data_indiv)
        data_hh_size['hhold_size'] = data_hh_size['hhold_size'].apply(lambda s: min(s,12))
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
        for v in varofint:
            var_name = '%s_div_hhold_size' % v
            data_hhold[var_name] = data_hhold[v]/data_hhold.hhold_size
        data_hhold.drop('hhold_size', axis=1, inplace=True)
    
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


# In[13]:

def read_test_train_v2():

    feat = dict()
    feat['mwi'] = dict()
    feat['mwi']['hh_size'] = True
    feat['mwi']['num_mean'] = True
    feat['mwi']['num_max'] = True
    feat['mwi']['num_min'] = True
    feat['mwi']['div_by_hh_size'] = True
    feat['mwi']['cat_hot'] = True    
    feat['mwi']['cat_hot_which'] =  []
    
    mwi_train = get_features(Country='mwi', f_dict=feat['mwi'], traintest='train')  
    mwi_test = get_features(Country='mwi', f_dict=feat['mwi'], traintest='test')  
   
    print("Country mwi")
    mwiX_train = pre_process_data(mwi_train.drop('poor', axis=1))
    mwiy_train = np.ravel(mwi_train.poor).astype(np.int8)

    # process the test data
    mwiX_test = pre_process_data(mwi_test, enforce_cols=mwiX_train.columns)
    
#     mwi_features = ['SlDKnCuu', 'jdetlNNF', 'maLAYXwi', 'vwpsXRGk', 'TYhoEiNm', 'zFkComtB', 'zzwlWZZC', 
#                  'DxLvCGgv', 'CbABToOI', 'qgMygRvX', 'uSKnVaKV', 'nzTeWUeM', 'nEsgxvAq', 'NmAVTtfA', 
#                  'YTdCRVJt', 'QyBloWXZ', 'HKMQJANN', 'ZRrposmO', 'HfKRIwMb', 'NRVuZwXK', 'UCAmikjV', 
#                  'UGbBCHRE', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 'ltcNxFzI', 'JwtIxvKg', 'bEPKkJXP', 
#                  'sFWbFEso', 'fHUZugEd', 'TqrXZaOw', 'galsfNtg', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT', 
#                  'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'GKUhYLAE', 
#                  'bIBQTaHw', 'KcArMKAe', 'enTUTSQi', 'wwfmpuWA', 'znHDEHZP', 'kWFVfHWP', 'HHAeIHna', 
#                  'dCGNTMiG', 'ngwuvaCV', 'XSgHIFXD', 'ANBCxZzU', 'NanLCXEI', 'SqEqFZsM', 'ZnBLVaqz',
#                  'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'udzhtHIr', 'IZFarbPw', 'lnfulcWk', 'QNLOXNwj', 
#                  'YFMZwKrU', 'RJQbcmKy', 'dlyiMEQt', 'TnWhKowI', 'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 
#                  'gfurxECf', 'hnrnuMte', 'XDDOZFWf', 'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 
#                  'DNAfxPzs', 'xkUFKUoW', 'jVDpuAmP', 'SeZULMCT', 'AtGRGAYi', 'WTFJilSZ', 'NBfffJUe', 
#                  'UXfyiodk', 'EftwspgZ', 'wKcZtLNv', 'szowPwNq', 'BfGjiYom', 'iWEFJYkR', 'BCehjxAl', 
#                  'nqndbwXP', 'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'tHFrzjai', 'zkbPtFyO', 
#                  'xZBEXWPR', 'dyGFeFAg', 'pKPTBZZq', 'bCYWWTxH', 'EQKKRGkR', 'muIetHMK', 'ishdUooQ', 
#                  'ItpCDLDM', 'gOGWzlYC', 'ptEAnCSs', 'HDCjCTRd', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts',
#                  'jwEuQQve', 'rQAsGegu', 'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 'TiwRslOh', 'PWShFLnY', 
#                  'lFExzVaF', 'IKqsuNvV', 'CqqwKRSn', 'YUExUvhq', 'yaHLJxDD', 'qlZMvcWc', 'ktBqxSwa', 
#                  'GIMIxlmv', 'wKVwRQIp', 'UaXLYMMh', 'bKtkhUWD', 'HhKXJWno', 'tAYCAXge', 'aWlBVrkK', 
#                  'cDkXTaWP', 'GHmAeUhZ', 'BIofZdtd', 'QZiSWCCB', 'CsGvKKBJ', 'JCDeZBXq', 'HGPWuGlV', 
#                  'nEsgxvAq_div_hhold_size', 'OMtioXZZ_div_hhold_size', 'YFMZwKrU_div_hhold_size', 
#                  'TiwRslOh_div_hhold_size', 'hhold_size', 'OdXpbPGJ', 'ukWqmeSS', 'ukWqmeSS_max', 
#                  'ukWqmeSS_min', 'kzSFB_ind_x', 'mOlYV_ind_x', 'axSTs_ind_x', 'YXCNt_ind_x', 'oArAw_ind_x', 
#                  'scxJu_ind_x', 'VzUws_ind_x', 'YwljV_ind_x', 'QkRds_ind_x', 'nUKzL_ind_x', 'OeQKE_ind_x', 
#                  'XNPgB_ind_x', 'dpMMl_ind_x', 'ndArQ_ind_x', 'GIApU_ind_x', 'Qydia_ind_x', 'vtkRP_ind_x',
#                  'sitaC_ind_x', 'VneGw_ind_x', 'rXEFU_ind_x', 'EAWFH_ind_x', 'UCsCT_ind_x', 'XQevi_ind_x', 
#                  'QQdHS_ind_x', 'uEstx_ind_x', 'Hikoa_ind_x', 'rkLqZ_ind_x', 'FUUXv_ind_x', 'juMSt_ind_x', 
#                  'SlRmt_ind_y', 'TRFeI_ind_y', 'dHZCo_ind_y', 'duBym_ind_y', 'lBMrM_ind_y', 'oGavK_ind_y', 
#                  'tMiQp_ind_y', 'wWIzo_ind_y', 'xnnDH_ind_y', 'yAyAe_ind_y', 'FRcdT_ind_y', 'UFoKR_ind_y',
#                  'CXizI_ind_y', 'JyIRx_ind_y', 'YsahA_ind_y', 'lzzev_ind_y', 'msICg_ind_y', 'NDnCs_ind_y', 
#                  'QyhRH_ind_y', 'XvoCa_ind_y', 'ccbZA_ind_y', 'fOUHD_ind_y', 'xMiWa_ind_y', 'bJTYb_ind_y', 
#                  'rwCRh_ind_y', 'scxJu_ind_y', 'OMzWB_ind_y', 'DgtXD_ind_y', 'EaHvf_ind_y', 'GmSKW_ind_y', 
#                  'VzUws_ind_y', 'uhOlG_ind_y', 'zfTDU_ind_y', 'IZbuU_ind_y', 'olfwp_ind_y', 'pdgUV_ind_y',
#                  'qIbMY_ind_y', 'sDvAm_ind_y', 'BQEnF_ind_y', 'Rjkzz_ind_y', 'VGNER_ind_y', 'bszTA_ind_y', 
#                  'xBZrP_ind_y', 'veBMo_ind_y', 'SowpV_ind_y', 'nUKzL_ind_y', 'OeQKE_ind_y', 'vSaJn_ind_y', 
#                  'CneHb_ind_y', 'JPCna_ind_y', 'MxNAc_ind_y', 'vvXmD_ind_y', 'TUafC_ind_y', 'dpMMl_ind_y', 
#                  'ndArQ_ind_y', 'zTqjB_ind_y', 'BNylo_ind_y', 'CXjLj_ind_y', 'AyuSE_ind_y', 'ZApCl_ind_y',
#                  'hCKQi_ind_y', 'Qydia_ind_y', 'vtkRP_ind_y', 'kVYrO_ind_y', 'VneGw_ind_y', 'rXEFU_ind_y', 
#                  'zncPX_ind_y', 'aKoLM_ind_y', 'DGyQh_ind_y', 'cEcbt_ind_y', 'xjHpn_ind_y', 'QBrMF_ind_y', 
#                  'mEGPl_ind_y', 'dAmhs_ind_y', 'gCSRj_ind_y', 'ESfgE_ind_y', 'Coacj_ind_y', 'dDnIb_ind_y', 
#                  'jVHyH_ind_y', 'rkLqZ_ind_y', 'xUYIC_ind_y', 'GtHel_ind_y', 'juMSt_ind_y']    
    
    # Refer to world-bank-ml-project/pover-t-tests/malawi/data/raw_mwi/Ordered%20Features%20Map.ipynb
    # for the mapping details
    mwi_features = ['cons_0305', 'cons_0408', 'farm_621', 'geo_district', 'cons_1336', 'gifts201', 'cons_1314',
                'cons_0302', 'farm_604', 'hld_lighting', 'cons_1104', 'farm_615', 'hld_nbcellpho', 'cons_1404',
                'hld_adeqcloth', 'cons_0801', 'farm_623', 'cons_0602', 'cons_0912', 'hld_walls', 'cons_0101',
                'com_schoolelec', 'cons_0504', 'cons_0202', 'cons_0824', 'com_roadtype', 'cons_0403', 'cons_0603',
                'cons_0413', 'cons_0905', 'cons_1405', 'cons_0113', 'cons_0404', 'cons_0703', 'hld_rubbish',
                'cons_0513', 'hld_credit1', 'cons_1103', 'hld_busin9', 'cons_1304', 'cons_1204', 'farm_616',
                'inc_102', 'cons_1303', 'cons_0911', 'cons_1324', 'cons_0503', 'cons_0303', 'hld_adeqfood',
                'own_528', 'cons_1308', 'hld_electricity', 'cons_0606', 'cons_0802', 'own_510', 'own_507',
                'farm_624', 'own_502', 'cons_0901', 'cons_0915', 'cons_0106', 'cons_0104', 'cons_1326',
                'hld_selfscale', 'own_501', 'own_513', 'cons_1214', 'cons_0111', 'cons_0204', 'cons_0501',
                'cons_1322', 'cons_1101', 'cons_1108', 'cons_1323', 'cons_1321', 'cons_1207', 'cons_0907',
                'cons_1420', 'cons_0803', 'cons_0702', 'com_publicphone', 'hld_dwelltype', 'cons_1109',
                'cons_1107', 'cons_0304', 'farm_617', 'cons_0511', 'cons_0108', 'hld_dwater', 'cons_1408',
                'cons_0301', 'cons_0821', 'cons_0112', 'cons_0410', 'farm_612', 'cons_0205', 'hld_cooking',
                'cons_0829', 'cons_1206', 'cons_1201', 'farm_619', 'cons_0203', 'com_weeklymrkt', 'farm_605',
                'com_classrooms', 'cons_0904', 'cons_0607', 'cons_0502', 'cons_1213', 'cons_1332', 'own_509',
                'cons_1419', 'cons_0506', 'cons_0913', 'cons_0102', 'farm_608', 'cons_0701', 'der_hhsize',
                'hld_foodsecurity', 'cons_0908', 'cons_0402', 'farm_620', 'cons_1217', 'farm_613', 'cons_0510',
                'cons_0505', 'cons_0508', 'com_clinic', 'hld_selfincome', 'cons_1409', 'cons_1302', 'cons_1328',
                'cons_0105', 'cons_1202', 'farm_607', 'hld_busin3', 'hld_nbguests', 'cons_0308', 'cons_1203',
                'cons_1305', 'hld_nbcellpho_div_hhold_size', 'hld_rooms_div_hhold_size',
                'hld_selfscale_div_hhold_size', 'der_hhsize_div_hhold_size', 'hhold_size', 'ind_work1', 'ind_age',
                'ind_age_max', 'ind_age_min', 'ind_rwenglish__Yes_ind_x', 'ind_rwenglish__No_ind_x',
                'ind_health5__No_ind_x', 'ind_health4__indiv_5_EMPTY_ind_x', 'ind_relation__Head_ind_x',
                'ind_educfath__NONE_ind_x', 'ind_educ03__indiv_11_EMPTY_ind_x',
                'ind_breakfast__indiv_13_EMPTY_ind_x', 'ind_health2__indiv_14_EMPTY_ind_x',
                'ind_educ07__indiv_15_EMPTY_ind_x', 'ind_religion__Christianity_ind_x',
                'ind_religion__Islam_ind_x', 'ind_educ08__No money for fees/ uniform_ind_x',
                'ind_educ08__indiv_18_EMPTY_ind_x', 'ind_birthattend__indiv_20_EMPTY_ind_x',
                'ind_sex__Female_ind_x', 'ind_sex__Male_ind_x', 'ind_birthplace__indiv_24_EMPTY_ind_x',
                'ind_educ05__No_ind_x', 'ind_educ05__indiv_25_EMPTY_ind_x', 'ind_rwchichewa__No_ind_x',
                'ind_educ02__No money for fees/uniform_ind_x', 'ind_readwrite__Yes_ind_x',
                'ind_educ12__indiv_31_EMPTY_ind_x', 'ind_work3__indiv_32_EMPTY_ind_x', 'ind_health1__No_ind_x',
                'ind_work2__No_ind_x', 'ind_educ01__No_ind_x', 'ind_educ01__Yes_ind_x', 'ind_educ06__std2_ind_y',
                'ind_educ06__Nursery/Pre school_ind_y', 'ind_educ06__std5_ind_y', 'ind_educ06__std3_ind_y',
                'ind_educ06__form4_ind_y', 'ind_educ06__std6_ind_y', 'ind_educ06__std7_ind_y',
                'ind_educ06__form3_ind_y', 'ind_educ06__std4_ind_y', 'ind_rwenglish__indiv_1_EMPTY_ind_y',
                'ind_health5__indiv_2_EMPTY_ind_y', 'ind_health5__Yes_ind_y', 'ind_language__indiv_3_EMPTY_ind_y',
                'ind_language__Lomwe_ind_y', 'ind_language__Tonga_ind_y', 'ind_language__Other_ind_y',
                'ind_language__Sena_ind_y', 'ind_educ10__Foot_ind_y', 'ind_relation__Grandchild_ind_y',
                'ind_relation__Niece/Nephew_ind_y', 'ind_relation__Wife/Husband_ind_y',
                'ind_relation__Child/Adopted child_ind_y', 'ind_educ11__Yes_ind_y',
                'ind_educfath__indiv_9_EMPTY_ind_y', 'ind_educfath__MSCE_ind_y', 'ind_educfath__NONE_ind_y',
                'ind_educ09__Pvt/Non religious_ind_y', 'ind_educ03__std3_ind_y', 'ind_educ03__std2_ind_y',
                'ind_educ03__std8_ind_y', 'ind_educ03__indiv_11_EMPTY_ind_y', 'ind_educ03__form4_ind_y',
                'ind_educ03__std1_ind_y', 'ind_marital__Widowed or widower_ind_y',
                'ind_marital__indiv_12_EMPTY_ind_y', 'ind_marital__Monogamous, married or non formal union_ind_y',
                'ind_marital__Never married_ind_y', 'ind_marital__Divorced_ind_y',
                'ind_breakfast__Tea/drink with solid food_ind_y', 'ind_breakfast__Milk/milk tea with sugar_ind_y',
                'ind_breakfast__Tea/drink with sugar_ind_y', 'ind_breakfast__Porridge with sugar_ind_y',
                'ind_breakfast__Porridge with g/nut flour_ind_y', 'ind_health2__Yes_ind_y', 'ind_educ07__No_ind_y',
                'ind_educ07__indiv_15_EMPTY_ind_y', 'ind_religion__Christianity_ind_y',
                'ind_religion__indiv_16_EMPTY_ind_y', 'ind_educ04__None_ind_y', 'ind_educ04__MSCE_ind_y',
                'ind_educ04__JCE_ind_y', 'ind_educ04__indiv_17_EMPTY_ind_y', 'ind_educ08__Found work_ind_y',
                'ind_educ08__No money for fees/ uniform_ind_y', 'ind_educ08__indiv_18_EMPTY_ind_y',
                'ind_educ08__Illness or disability_ind_y', 'ind_health3__Yes_ind_y',
                'ind_health3__indiv_19_EMPTY_ind_y', 'ind_educmoth__indiv_21_EMPTY_ind_y',
                'ind_educmoth__PSLC_ind_y', 'ind_educmoth__NONE_ind_y', 'ind_sex__Female_ind_y',
                'ind_sex__Male_ind_y', 'ind_health8__Na (if not working or not attending schoool)_ind_y',
                'ind_educ05__No_ind_y', 'ind_educ05__indiv_25_EMPTY_ind_y', 'ind_educ05__Yes_ind_y',
                'ind_rwchichewa__indiv_26_EMPTY_ind_y', "ind_educ02__Parents didn't let me in_ind_y",
                'ind_educ02__Had to work or help at home_ind_y', 'ind_educ02__Not interested/lazy_ind_y',
                'ind_work6__Yes_ind_y', 'ind_work6__indiv_28_EMPTY_ind_y', 'ind_work3__Yes_ind_y',
                'ind_work3__No_ind_y', 'ind_health1__indiv_33_EMPTY_ind_y',
                'ind_work5__State-Owned Enterprise (Parastatal)_ind_y', 'ind_work5__Private Company_ind_y',
                'ind_work2__indiv_35_EMPTY_ind_y', 'ind_work2__No_ind_y', 'ind_work2__Yes_ind_y',
                'ind_educ01__indiv_36_EMPTY_ind_y', 'ind_educ01__Yes_ind_y']
    
    mwiX_train =  mwiX_train[mwi_features].copy()
    mwiX_test =  mwiX_test[mwi_features].copy()
    print("--------------------------------------------")
    return mwiX_train, mwiy_train, mwiX_test


# In[14]:

mwiX_train, mwiY_train, mwiX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[15]:

models = {'mwi':'Keras_mwi01'}

datafiles = {}
datafiles['out'] = 'predictions/KerasUB_M03_F11_'


# ## Submission

# In[16]:

mwi_preds = Bagging_Test(mwiX_train, mwiY_train, mwiX_test, 'mwi')


# In[17]:

# convert preds to data frames
mwi_sub = make_country_df(mwi_preds.flatten(), mwiX_test, 'mwi')


# In[18]:

mwi_sub.to_csv(datafiles['out']+'_mwi_test.csv')


# In[ ]:



