
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
    country_sub = pd.DataFrame(data=preds,  # proba p=1
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
    gbm = lgb.train(params,lgb_train,categorical_feature=cat_list,num_boost_round=3200)


    Yt = gbm.predict(Xte)
    return Yt

# # Data Processing

# In[5]:

data_paths = {
    'mwi': {
        'train_hhold': '../../data/raw_mwi/mwi_aligned_hhold_train.csv',
        'test_hhold':  '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
        'train_indiv': '../../data/raw_mwi/mwi_aligned_indiv_train.csv',
        'test_indiv':  '../../data/raw_mwi/mwi_aligned_indiv_test.csv'
    }
}


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
 
    aux = ~data_indiv.drop(var2drop, axis=1).dtypes.isin(['object','bool','O'])
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


# In[6]:

def get_features(Country='mwi', f_dict=None, traintest='train'):
      
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


# In[7]:

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


# In[8]:

def read_test_train_v2():

    feat = dict()
    feat['mwi'] = dict()
    feat['mwi']['hh_size'] = True
    feat['mwi']['num_mean'] = True
    feat['mwi']['num_max'] = True
    feat['mwi']['num_min'] = True
    feat['mwi']['cat_hot'] = True
    feat['mwi']['cat_hot_which'] =  []
    
    mwi_train = get_features(Country='mwi', f_dict=feat['mwi'], traintest='train')  
    mwi_test = get_features(Country='mwi', f_dict=feat['mwi'], traintest='test')  
   
    print("Country mwi")
    mwiX_train = pre_process_data(mwi_train.drop('poor', axis=1))
    mwiy_train = np.ravel(mwi_train.poor).astype(np.int8)

    # process the test data
    mwiX_test = pre_process_data(mwi_test, enforce_cols=mwiX_train.columns)

#     mwi_features = ['SlDKnCuu', 'jdetlNNF', 'vwpsXRGk', 'TYhoEiNm', 'VZtBaoXL', 'zFkComtB', 'zzwlWZZC', 
#                  'DxLvCGgv', 'CbABToOI', 'qgMygRvX', 'uSKnVaKV', 'nEsgxvAq', 'NmAVTtfA', 'YTdCRVJt', 
#                  'QyBloWXZ', 'HKMQJANN', 'ZRrposmO', 'EJgrQqET', 'HfKRIwMb', 'NRVuZwXK', 'UCAmikjV', 
#                  'UGbBCHRE', 'uJYGhXqG', 'bxKGlBYX', 'nCzVgxgY', 'MxOgekdE', 'SqGRfEuW', 'JwtIxvKg', 
#                  'bEPKkJXP', 'cqUmYeAp', 'sFWbFEso', 'TqrXZaOw', 'VIRwrkXp', 'gwhBRami', 'bPOwgKnT', 
#                  'fpHOwfAs', 'VXXLUaXP', 'btgWptTG', 'YWwNfVtR', 'bgoWYRMQ', 'bMudmjzJ', 'GKUhYLAE',
#                  'OMtioXZZ', 'bIBQTaHw', 'KcArMKAe', 'enTUTSQi', 'wwfmpuWA', 'znHDEHZP', 'kWFVfHWP', 
#                  'XwVALSPR', 'CrfscGZl', 'dCGNTMiG', 'ngwuvaCV', 'XSgHIFXD', 'ANBCxZzU', 'NanLCXEI', 
#                  'ZnBLVaqz', 'srPNUgVy', 'pCgBHqsR', 'wEbmsuJO', 'TWXCrjor', 'mRgnuJVE', 'pWyRKfsb', 
#                  'udzhtHIr', 'IZFarbPw', 'QNLOXNwj', 'YFMZwKrU', 'RJQbcmKy', 'TnWhKowI', 'LoYIbglA', 
#                  'GhJKwVWC', 'lVHmBCmb', 'qgxmqJKa', 'gfurxECf', 'hnrnuMte', 'XDDOZFWf', 'ccAHraiP', 
#                  'QayGNSmS', 'ePtrWTFd', 'tbsBPHFD', 'naDKOzdk', 'DNAfxPzs', 'xkUFKUoW', 'SeZULMCT', 
#                  'AtGRGAYi', 'FGDcbVBN', 'WTFJilSZ', 'NBfffJUe', 'mvgxfsRb', 'UXfyiodk', 'EftwspgZ', 
#                  'bSaLisbO', 'wKcZtLNv', 'BfGjiYom', 'iWEFJYkR', 'BCehjxAl', 'CHAQHqqr', 'nqndbwXP',
#                  'phwExnuQ', 'SzUcfjnr', 'PXtHzrqw', 'CNkSTLvx', 'MKozKLvT', 'zkbPtFyO', 'xZBEXWPR', 
#                  'dyGFeFAg', 'bCYWWTxH', 'EQKKRGkR', 'muIetHMK', 'ItpCDLDM', 'gOGWzlYC', 'ptEAnCSs', 
#                  'HDCjCTRd', 'orfSPOJX', 'OKMtkqdQ', 'qTginJts', 'JzhdOhzb', 'jwEuQQve', 'rQAsGegu', 
#                  'kLkPtNnh', 'CtHqaXhY', 'FmSlImli', 'TiwRslOh', 'PWShFLnY', 'lFExzVaF', 'IKqsuNvV',
#                  'CqqwKRSn', 'YUExUvhq','yaHLJxDD', 'qlZMvcWc', 'dqRtXzav', 'ktBqxSwa', 'NqPjMmKP',
#                  'GIMIxlmv', 'UaXLYMMh', 'bKtkhUWD', 'HhKXJWno', 'tAYCAXge', 'WAFKMNwv', 'aWlBVrkK', 
#                  'cDkXTaWP', 'hnmsRSvN', 'GHmAeUhZ', 'BIofZdtd', 'QZiSWCCB', 'CsGvKKBJ', 'OLpGAaEu', 
#                  'JCDeZBXq', 'WuwrCsIY', 'AlDbXTlZ', 'hhold_size', 'OdXpbPGJ', 'ukWqmeSS', 'ukWqmeSS_max', 
#                  'ukWqmeSS_min', 'mOlYV_ind_x', 'JyIRx_ind_x', 'msICg_ind_x', 'YXCNt_ind_x', 'oArAw_ind_x', 
#                  'HgfUG_ind_x', 'tqINY_ind_x', 'EaHvf_ind_x', 'GmSKW_ind_x', 'pdgUV_ind_x', 'xrEKh_ind_x', 
#                  'QkRds_ind_x', 'TGbFh_ind_x', 'veIDf_ind_x', 'vvXmD_ind_x', 'ndArQ_ind_x', 'KOjYm_ind_x', 
#                  'hCKQi_ind_x', 'Qydia_ind_x', 'vtkRP_ind_x', 'EAWFH_ind_x', 'xjHpn_ind_x', 'RPBUw_ind_x', 
#                  'yOwsR_ind_x', 'dAmhs_ind_x', 'uEstx_ind_x', 'OkXob_ind_x', 'zQvdC_ind_x', 'juMSt_ind_x', 
#                  'JTCKs_ind_x', 'SlRmt_ind_y', 'TRFeI_ind_y', 'dHZCo_ind_y', 'duBym_ind_y', 'oGavK_ind_y', 
#                  'tMiQp_ind_y', 'wWIzo_ind_y', 'mOlYV_ind_y', 'CXizI_ind_y', 'DQhEE_ind_y','HIvIU_ind_y', 
#                  'JyIRx_ind_y', 'LvUxT_ind_y','YsahA_ind_y', 'AvBOo_ind_y', 'BqqGq_ind_y', 'QyhRH_ind_y', 
#                  'ccbZA_ind_y', 'fOUHD_ind_y', 'pWLuE_ind_y', 'kpkiH_ind_y', 'rwCRh_ind_y', 'OMzWB_ind_y',
#                  'Whopv_ind_y', 'cHNSE_ind_y', 'zCwHm_ind_y', 'AYcgs_ind_y', 'DgtXD_ind_y', 'EaHvf_ind_y', 
#                  'GmSKW_ind_y', 'pRitH_ind_y', 'yhUHu_ind_y', 'zfTDU_ind_y', 'kzJXk_ind_y', 'pdgUV_ind_y', 
#                  'qIbMY_ind_y', 'sDvAm_ind_y', 'xrEKh_ind_y', 'bszTA_ind_y', 'xBZrP_ind_y', 'veBMo_ind_y', 
#                  'SowpV_ind_y', 'OeQKE_ind_y', 'XNPgB_ind_y', 'veIDf_ind_y', 'MxNAc_ind_y', 'SuzRU_ind_y',
#                  'PaHYu_ind_y', 'SjaWF_ind_y', 'TUafC_ind_y', 'dpMMl_ind_y', 'meQRz_ind_y', 'zTqjB_ind_y',
#                  'BNylo_ind_y', 'CXjLj_ind_y', 'PwkMV_ind_y', 'GxyHv_ind_y', 'PrZhn_ind_y', 'ZApCl_ind_y',
#                  'hCKQi_ind_y', 'Qydia_ind_y', 'vtkRP_ind_y', 'kVYrO_ind_y', 'OoqEw_ind_y', 'SWhXf_ind_y', 
#                  'UCsCT_ind_y', 'uJdwX_ind_y', 'QBrMF_ind_y', 'mEGPl_ind_y', 'qmOVd_ind_y', 'yOwsR_ind_y', 
#                  'Jarbl_ind_y', 'dAmhs_ind_y', 'ESfgE_ind_y', 'okwnE_ind_y', 'xUYIC_ind_y', 'GtHel_ind_y', 
#                  'vhhVz_ind_y']

    # Refer to world-bank-ml-project/pover-t-tests/malawi/data/raw_mwi/Ordered%20Features%20Map.ipynb
    # for the mapping details
    max_features = ['cons_0305', 'cons_0408', 'geo_district', 'cons_1336', 'inc_109', 'gifts201', 'cons_1314',
                'cons_0302', 'farm_604', 'hld_lighting', 'cons_1104', 'hld_nbcellpho', 'cons_1404',
                'hld_adeqcloth', 'cons_0801', 'farm_623', 'cons_0602', 'cons_1220', 'cons_0912', 'hld_walls',
                'cons_0101', 'com_schoolelec', 'cons_0504', 'cons_0202', 'cons_0824', 'cons_1417',
                'hld_adeqhealth', 'cons_0403', 'cons_0603', 'hld_dwelloccu', 'cons_0413', 'cons_1405', 'cons_0404',
                'cons_0703', 'hld_rubbish', 'cons_0513', 'hld_credit1', 'cons_1103', 'hld_busin9', 'cons_1304',
                'cons_1204', 'farm_616', 'hld_rooms', 'inc_102', 'cons_1303', 'cons_0911', 'cons_1324',
                'cons_0503', 'cons_0303', 'cons_1310', 'hld_bednet', 'own_528', 'cons_1308', 'hld_electricity',
                'cons_0606', 'cons_0802', 'own_507', 'farm_624', 'own_502', 'cons_0901', 'inc_113', 'inc_115',
                'com_bank', 'cons_0915', 'cons_0106', 'cons_1326', 'hld_selfscale', 'own_501', 'cons_1214',
                'cons_0109', 'cons_0111', 'cons_0204', 'cons_0501', 'cons_1322', 'cons_1101', 'cons_1108',
                'cons_1320', 'cons_1323', 'cons_1321', 'cons_1207', 'cons_0907', 'cons_1420', 'cons_0803',
                'com_publicphone', 'hld_dwelltype', 'hld_busin4', 'cons_1109', 'cons_1107', 'com_bus', 'cons_0304',
                'farm_617', 'cons_1219', 'cons_0511', 'hld_dwater', 'cons_1408', 'cons_0301', 'cons_0916',
                'cons_0821', 'cons_0112', 'cons_0410', 'farm_612', 'cons_0205', 'cons_0812', 'cons_0829',
                'cons_1206', 'cons_1201', 'cons_0203', 'com_weeklymrkt', 'farm_605', 'cons_0904', 'cons_0607',
                'cons_0502', 'cons_1213', 'cons_1332', 'own_509', 'cons_1419', 'cons_0207', 'cons_0506',
                'cons_0913', 'cons_0102', 'farm_608', 'cons_0701', 'der_hhsize', 'hld_foodsecurity', 'cons_0908',
                'cons_0402', 'farm_620', 'cons_1217', 'farm_613', 'cons_0510', 'cons_0601', 'cons_0505', 'own_512',
                'cons_0508', 'hld_selfincome', 'cons_1409', 'cons_1302', 'cons_1328', 'hld_busin1', 'cons_0105',
                'cons_1202', 'cons_0826', 'farm_607', 'hld_busin3', 'hld_nbguests', 'cons_0308', 'farm_606',
                'cons_1203', 'own_525', 'com_dailymrkt', 'hhold_size', 'ind_work1', 'ind_age', 'ind_age_max',
                'ind_age_min', 'ind_rwenglish__No_ind_x', 'ind_language__Lomwe_ind_x', 'ind_language__Sena_ind_x',
                'ind_health4__indiv_5_EMPTY_ind_x', 'ind_relation__Head_ind_x', 'ind_health7__indiv_7_EMPTY_ind_x',
                'ind_educfath__PSLC_ind_x', 'ind_educ03__std2_ind_x', 'ind_educ03__std8_ind_x',
                'ind_marital__Monogamous, married or non formal union_ind_x', 'ind_marital__Separated_ind_x',
                'ind_health2__indiv_14_EMPTY_ind_x', 'ind_religion__Traditional_ind_x', 'ind_religion__None_ind_x',
                'ind_educ04__indiv_17_EMPTY_ind_x', 'ind_educ08__indiv_18_EMPTY_ind_x', 'ind_health3__No_ind_x',
                'ind_educmoth__NONE_ind_x', 'ind_sex__Female_ind_x', 'ind_sex__Male_ind_x',
                'ind_rwchichewa__No_ind_x', 'ind_educ02__Not interested/lazy_ind_x', 'ind_work4__Yes_ind_x',
                'ind_work4__No_ind_x', 'ind_work3__Yes_ind_x', 'ind_work3__indiv_32_EMPTY_ind_x',
                'ind_work5__Private Individual_ind_x', 'ind_work5__indiv_34_EMPTY_ind_x', 'ind_educ01__Yes_ind_x',
                'ind_health6__indiv_37_EMPTY_ind_x', 'ind_educ06__std2_ind_y',
                'ind_educ06__Nursery/Pre school_ind_y', 'ind_educ06__std5_ind_y', 'ind_educ06__std3_ind_y',
                'ind_educ06__std6_ind_y', 'ind_educ06__std7_ind_y', 'ind_educ06__form3_ind_y',
                'ind_rwenglish__No_ind_y', 'ind_language__indiv_3_EMPTY_ind_y', 'ind_language__Tumbuka_ind_y',
                'ind_language__Chewa_ind_y', 'ind_language__Lomwe_ind_y', 'ind_language__Nyanja_ind_y',
                'ind_language__Tonga_ind_y', 'ind_health4__No_ind_y', 'ind_relation__Father/Mother_ind_y',
                'ind_relation__Grandchild_ind_y', 'ind_relation__Wife/Husband_ind_y',
                'ind_relation__Child/Adopted child_ind_y', 'ind_relation__Sister/Brother_ind_y',
                'ind_health7__Na (if not working or not attending schoool)_ind_y', 'ind_educfath__MSCE_ind_y',
                'ind_educ09__Pvt/Non religious_ind_y', 'ind_educ09__Secondary - Church/ mission school_ind_y',
                'ind_educ09__Gvt(Conventional)_ind_y', 'ind_educ09__Community day(CDSS)_ind_y',
                'ind_educ03__std7_ind_y', 'ind_educ03__std3_ind_y', 'ind_educ03__std2_ind_y',
                'ind_educ03__std8_ind_y', 'ind_educ03__Nursery/Pre school_ind_y', 'ind_educ03__std5_ind_y',
                'ind_educ03__std1_ind_y', 'ind_marital__Polygamous,married or non formal union_ind_y',
                'ind_marital__Monogamous, married or non formal union_ind_y', 'ind_marital__Never married_ind_y',
                'ind_marital__Divorced_ind_y', 'ind_marital__Separated_ind_y',
                'ind_breakfast__Porridge with sugar_ind_y', 'ind_breakfast__Porridge with g/nut flour_ind_y',
                'ind_health2__Yes_ind_y', 'ind_educ07__No_ind_y', 'ind_religion__Christianity_ind_y',
                'ind_religion__Islam_ind_y', 'ind_religion__None_ind_y', 'ind_educ04__JCE_ind_y',
                'ind_educ04__PSLC_ind_y', 'ind_educ08__Acquired all education wanted_ind_y',
                'ind_educ08__Not interested, lazy_ind_y', 'ind_educ08__Found work_ind_y',
                'ind_educ08__No money for fees/ uniform_ind_y', 'ind_educ08__Too old to continue_ind_y',
                'ind_educ08__Illness or disability_ind_y', 'ind_health3__Yes_ind_y',
                'ind_health3__indiv_19_EMPTY_ind_y', 'ind_birthattend__TBA_ind_y', 'ind_educmoth__JCE_ind_y',
                'ind_educmoth__NON-UNIV. DIPLOMA_ind_y', 'ind_educmoth__PSLC_ind_y', 'ind_educmoth__NONE_ind_y',
                'ind_sex__Female_ind_y', 'ind_sex__Male_ind_y',
                'ind_health8__Na (if not working or not attending schoool)_ind_y',
                'ind_educ02__School conflict with beliefs_ind_y', 'ind_educ02__School too far from Home_ind_y',
                'ind_educ02__No money for fees/uniform_ind_y',
                'ind_educ02__Still too young to attend school_ind_y', 'ind_work6__Yes_ind_y',
                'ind_work6__indiv_28_EMPTY_ind_y', 'ind_work6__No_ind_y', 'ind_work4__No_ind_y',
                'ind_educ12__Other (specify)_ind_y', 'ind_work3__Yes_ind_y', 'ind_health1__indiv_33_EMPTY_ind_y',
                'ind_health1__Yes_ind_y', 'ind_work2__Yes_ind_y', 'ind_educ01__indiv_36_EMPTY_ind_y',
                'ind_health6__No_ind_y']

    mwiX_train =  mwiX_train[max_features].copy()
    mwiX_test =  mwiX_test[max_features].copy()
    print("--------------------------------------------")
    return mwiX_train, mwiy_train, mwiX_test

# In[9]:

mwiX_train, mwiY_train, mwiX_test = read_test_train_v2()


# # Model Train/Predict

# ## Def

# In[10]:

model = {'mwi':'model_mwi_v1'}

datafiles = {}
datafiles['out'] = 'predictions/Light_M01_F10_'


# ## Submission

# In[11]:

mwi_preds = eval(model['mwi'])(mwiX_train, mwiY_train, mwiX_test)


# In[12]:

# convert preds to data frames
mwi_sub = make_country_df(mwi_preds.flatten(), mwiX_test, 'mwi')


# In[13]:

mwi_sub.to_csv(datafiles['out']+'_mwi_test.csv')


# In[ ]:



