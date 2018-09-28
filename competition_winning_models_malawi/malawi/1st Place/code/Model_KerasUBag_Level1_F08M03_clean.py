
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

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


# In[7]:

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

# In[8]:

def expand_dims(x):
    return K.expand_dims(x, 1)

def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])


# In[9]:

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
        Dout[col] = max(3, min(50, (cat_sz+1)//2))
    
    df_train = df_all.iloc[:ntrain,:].copy()
    df_test = df_all.iloc[ntrain:,:].copy()
    return df_train,df_test, num_list, cat_list, Din, Dout


# In[10]:

def Keras_mwi01(Xtr,Ytr,Xte):
    
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
    
    model.fit(X_train, Ytr, batch_size=256, epochs=9, verbose=0, shuffle=True)
 
    Yt = model.predict(X_test).flatten() 
    K.clear_session()
    return Yt


# In[11]:

def Bagging_Test(Xtr, Ytr, Xte,c):
    Yt_av =  np.zeros(Xte.shape[0])
    nbags = 3
    nfolds = 8
    kf = 0
    for i in range(nfolds):
        xtr, ytr = resample(Xtr,Ytr,n_samples=int(0.95*Xtr.shape[0]),replace=False,random_state=10*i)
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

# In[12]:

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


# In[13]:

data_paths = {
    'mwi': {
        'train_hhold': '../../data/raw_mwi/mwi_aligned_hhold_train.csv',
        'test_hhold':  '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
        'train_indiv': '../../data/raw_mwi/mwi_aligned_indiv_train.csv',
        'test_indiv':  '../../data/raw_mwi/mwi_aligned_indiv_test.csv'
    }
}

# In[14]:

def get_hhold_size(data_indiv):
    return data_indiv.groupby('id').country.agg({'hhold_size':'count'})


# In[15]:

def get_features(Country='mwi', f_dict=None, traintest='train'):
      
    # load data
    data_hhold = pd.read_csv(data_paths[Country]['%s_hhold' % traintest], index_col='id')
    data_indiv = pd.read_csv(data_paths[Country]['%s_indiv' % traintest], index_col='id')

    ## Add indiv features:
    #hhold size
    if f_dict.get('hh_size'):
        data_hh_size = get_hhold_size(data_indiv)
        data_hhold = data_hhold.merge(data_hh_size, left_index=True, right_index=True)
    
    return data_hhold


# In[16]:

def read_test_train_v2():

    feat = dict()
    feat['mwi'] = dict()
    feat['mwi']['hh_size'] = True
    
    mwi_train = get_features(Country='mwi', f_dict=feat['mwi'], traintest='train')  
    mwi_test = get_features(Country='mwi', f_dict=feat['mwi'], traintest='test')    

    print("Country mwi")
    mwiX_train = pre_process_data(mwi_train.drop('poor', axis=1))
    mwiy_train = np.ravel(mwi_train.poor)

    # process the test data
    mwiX_test = pre_process_data(mwi_test, enforce_cols=mwiX_train.columns)

#     aremove_list = ['KAJOWiiw', 'DsKacCdL', 'rtPrBBPl', 'tMJrvvut', 'TYhoEiNm',
#                     'bgfNZfcj', 'sYIButva', 'VZtBaoXL', 'GUvFHPNA', 'fxbqfEWb',
#                     'nGTepfos', 'CbABToOI', 'uSKnVaKV', 'hESBInAl', 'BbKZUYsB',
#                     'UCnazcxd', 'aCfsveTu', 'EfkPrfXa', 'FcekeISI', 'wakWLjkG',
#                     'dkoIJCbY', 'NrUWfvEq', 'WqhniYIc', 'IIEHQNUc', 'UGbBCHRE',
#                     'bxKGlBYX', 'MxOgekdE', 'ggNglVqE', 'YDgWYWcJ', 'SqGRfEuW',
#                     'benRXROb', 'dSALvhyd', 'gfmfEyjQ', 'WbxAxHul', 'FlBqizNL',
#                     'KjkrfGLD', 'JbjHTYUM', 'HmDAlkAH', 'galsfNtg', 'dsIjcEFe',
#                     'OybQOufM', 'ihGjxdDj', 'FGYOIJbC', 'UBanubTh', 'NIRMacrk',
#                     'wxDnGIwN', 'rAkSnhJF', 'glEjrMIg', 'GKUhYLAE', 'SsZAZLma',
#                     'KcArMKAe', 'TFrimNtw', 'LjvKYNON', 'wwfmpuWA', 'TvShZEBA',
#                     'nuwxPLMe', 'eeYoszDM', 'HHAeIHna', 'CrfscGZl', 'SqEqFZsM',
#                     'lFcfBRGd', 'AsEmHUzj', 'pyBSpOoN', 'srPNUgVy', 'TWXCrjor',
#                     'wgWdGBOp', 'ErggjCIN', 'lnfulcWk', 'UHGnBrNt', 'QNLOXNwj',
#                     'ytYMzOlW', 'ucXrHdoC', 'iBQXwnGC', 'sslNoPlw', 'InULRrrv',
#                     'LoYIbglA', 'EuJrVjyG', 'nSzbETYS', 'CpqWSQcW', 'XqURHMoh',
#                     'mDTcQhdH', 'mvGdZZcs', 'CbzSWtkF', 'LrQXqVUj', 'CIGUXrRQ',
#                     'CtFxPQPT', 'ePtrWTFd', 'lTAXSTys', 'dEpQghsA', 'SeZULMCT',
#                     'NitzgUzY', 'YlZCqMNw', 'rYvVKPAF', 'rfDBJtIz', 'KHzKOKPw',
#                     'EftwspgZ', 'mycoyYwl', 'ySkAFOzx', 'dkPWxwSF', 'bSaLisbO',
#                     'wKcZtLNv', 'mBlWbDmc', 'szowPwNq', 'ULMvnWcn', 'ogHwwdzc',
#                     'uZGqTQUP', 'PXtHzrqw', 'MKozKLvT', 'zkbPtFyO', 'HfOrdgBo',
#                     'YKwvJgoP', 'rnJOTwVD', 'xNUUjCIL', 'JMNvdasy', 'MBQcYnjc',
#                     'cCsFudxF', 'hJrMTBVd', 'ishdUooQ', 'gOGWzlYC', 'HDCjCTRd',
#                     'lOujHrCk', 'MARfVwUE', 'orfSPOJX', 'QBJeqwPF', 'JzhdOhzb',
#                     'THDtJuYh', 'nKoaotpH', 'TzPdCEPV', 'DbUNVFwv', 'UsENDgsH',
#                     'PWShFLnY', 'uRFXnNKV', 'CVCsOVew', 'tlxXCDiW', 'CqqwKRSn',
#                     'YUExUvhq', 'UXhTXbuS', 'yaHLJxDD', 'zuMWFXax', 'ALbGNqKm',
#                     'tOWnWxYe', 'RvTenIlS', 'wKVwRQIp', 'ncjMNgfp', 'RJFKdmYJ',
#                     'gllMXToa', 'VFTkSOrq', 'WAFKMNwv', 'mHzqKSuN', 'UjuNwfjv',
#                     'cDkXTaWP', 'GHmAeUhZ', 'VBjVVDwp', 'kZVpcgJL', 'sDGibZrP',
#                     'OLpGAaEu', 'LrDrWRjC', 'AlDbXTlZ']
    
    # Refer to world-bank-ml-project/pover-t-tests/malawi/data/raw_mwi/Ordered%20Features%20Map.ipynb
    # for the mapping details
    mwi_remove_list = ['inc_112', 'cons_1205', 'cons_1329', 'cons_0306', 'cons_1336', 'cons_0103', 'cons_0810', 'inc_109',
                    'cons_0115', 'cons_1311', 'cons_1309', 'farm_604', 'cons_1104', 'inc_111', 'hld_busin2',
                    'com_postoffice', 'cons_0610', 'own_505', 'farm_618', 'own_516', 'cons_0910', 'inc_105',
                    'cons_0514', 'cons_1330', 'com_schoolelec', 'cons_0202', 'cons_1417', 'cons_0401', 'own_519',
                    'hld_adeqhealth', 'cons_0507', 'cons_1338', 'hld_toiletshr', 'cons_0309', 'cons_0201',
                    'com_distclinic', 'farm_622', 'inc_110', 'cons_0113', 'cons_0414', 'cons_0411', 'cons_0827',
                    'own_532', 'farm_601', 'cons_1331', 'cons_1325', 'com_distprimary', 'cons_1102', 'farm_616',
                    'cons_0608', 'cons_1303', 'cons_1306', 'hld_floor', 'cons_1324', 'cons_0814', 'cons_1315',
                    'cons_0409', 'hld_adeqfood', 'hld_bednet', 'own_510', 'own_522', 'hld_headsleep', 'cons_0515',
                    'farm_624', 'inc_113', 'own_511', 'cons_1319', 'cons_0104', 'inc_106', 'cons_1326', 'cons_1327',
                    'inc_101', 'cons_1105', 'cons_1301', 'cons_1410', 'cons_0109', 'own_503', 'inc_114', 'own_514',
                    'own_517', 'hld_busin5', 'inc_104', 'cons_0902', 'cons_0816', 'own_504', 'cons_0117', 'cons_1321',
                    'cons_1210', 'cons_1416', 'com_publicphone', 'cons_0706', 'gifts202', 'cons_1402', 'farm_611',
                    'cons_0811', 'farm_617', 'cons_1318', 'farm_610', 'hld_toilet', 'cons_1219', 'cons_0511',
                    'cons_1312', 'cons_0108', 'com_urbancenter', 'cons_0407', 'farm_609', 'farm_612', 'cons_0812',
                    'cons_0829', 'cons_1406', 'geo_urbrur', 'cons_0707', 'cons_0818', 'farm_614', 'cons_0107',
                    'own_529', 'cons_1334', 'com_classrooms', 'cons_0607', 'cons_1213', 'cons_1418', 'cons_0605',
                    'cons_1332', 'cons_0828', 'cons_0207', 'cons_0914', 'cons_1216', 'cons_0310', 'cons_1211',
                    'com_medicines', 'hld_foodsecurity', 'hld_adeqhous', 'cons_0705', 'own_523', 'farm_620',
                    'cons_1217', 'gifts203', 'farm_613', 'inc_108', 'cons_0820', 'cons_0709', 'cons_1411',
                    'com_clinic', 'own_524', 'cons_1403', 'farm_603', 'own_531', 'hld_busin1', 'cons_0825',
                    'hld_dwateros', 'cons_1202', 'farm_607', 'farm_602', 'cons_0406', 'cons_0209', 'farm_606',
                    'cons_1106', 'com_dailymrkt']

    mwiX_train.drop(mwi_remove_list, axis=1, inplace=True)
    mwiX_test.drop(mwi_remove_list, axis=1, inplace=True)
    
    print("--------------------------------------------")
    return mwiX_train, mwiy_train, mwiX_test


# In[17]:

mwiX_train, mwiY_train, mwiX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[18]:

models = {'mwi':'Keras_mwi01'}

datafiles = {}
datafiles['out'] = 'predictions/KerasUB_M03_F08_'


# ## Submission

# In[19]:

mwi_preds = Bagging_Test(mwiX_train, mwiY_train, mwiX_test,'mwi')


# In[20]:

# convert preds to data frames
mwi_sub = make_country_df(mwi_preds.flatten(), mwiX_test, 'mwi')


# In[21]:

mwi_sub.to_csv(datafiles['out']+'_mwi_test.csv')


# In[ ]:



