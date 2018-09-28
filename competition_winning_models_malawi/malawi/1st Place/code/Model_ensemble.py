
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
random_state = np.random.RandomState(2925)
np.random.seed(2925) # for reproducibility


# The ensemble weights were obtained via CV with the **minimize** function of **scipy.optimize** library

# # Functions

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


# In[3]:

def fopt_pred(pars, Ym):
    pars = pars/pars.sum()
    Y=np.dot(np.concatenate(Ym,axis=1),np.atleast_2d(pars).T)
    return Y


# # Data Processing

# In[4]:

datafiles = {}
datafiles['out'] = 'predictions/final_submission'


# In[5]:

def read_test_train():

    data_paths = {
        'mwi': {
            'train': '../../data/raw_mwi/mwi_aligned_hhold_train.csv',
            'test':  '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
        }
    }
    
    # load training data
    mwi_train = pd.read_csv(data_paths['mwi']['train'], index_col='id')

    mwiy_train = np.ravel(mwi_train.poor)
    mwiX_train = mwi_train['country'].copy()
    
    # load test data
    mwi_test = pd.read_csv(data_paths['mwi']['test'], index_col='id')
    # process the test data
    mwiX_test = mwi_test['country'].copy()
    return mwiX_train, mwiy_train, mwiX_test


# In[6]:

mwiX_train, mwiY_train, mwiX_test = read_test_train()


# # mwi

# In[9]:

models_mwi = ['predictions/Keras_M03_F09','predictions/KerasUB_M03_F11','predictions/KerasUB_M02_F02','predictions/KerasUB_M03_F02','predictions/KerasUB_M03_F08','predictions/Light_M01_F08','predictions/Light_M01_F09','predictions/Light_M01_F10','predictions/Light_M01_F11']

mwiYp_test = []

for file in models_mwi:
    print(file)
    testd = pd.read_csv(file + '__mwi_test.csv')
    mwiYp_test.append(np.atleast_2d(testd[testd['country']=='mwi']['poor'].values).T)


# # Submission

# In[16]:

pcv_mwi = np.array([1.54650548e-01, 6.99449920e-02, 2.18102764e-17, 7.77701279e-03, 2.96078796e-01, 1.53370409e-01, 1.31249388e-01, 8.78751040e-02, 9.90537502e-02])


# In[17]:

mwiYt_pred =  fopt_pred(pcv_mwi, mwiYp_test)


# In[18]:

# convert preds to data frames
mwi_sub = make_country_df(mwiYt_pred.flatten(), mwiX_test, 'mwi')


# In[54]:

submission = mwi_sub

# In[20]:

submission.to_csv(datafiles['out']+'test.csv')
