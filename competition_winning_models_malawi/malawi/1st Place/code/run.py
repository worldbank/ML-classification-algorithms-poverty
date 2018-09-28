import os

print('running\n')

notebook_list = [
                 'python Model_LightGBM_Level1_F11M01_clean.py',
                 'python Model_LightGBM_Level1_F10M01_clean.py',
                 'python Model_LightGBM_Level1_F09M01_clean.py',
                 'python Model_LightGBM_Level1_F08M01_clean.py',
                 'python Model_Keras_Level1_F09M03_clean.py',
                 'python Model_KerasUBag_Level1_F11M03_clean.py',
                 'python Model_KerasUBag_Level1_F08M03_clean.py',
                 'python Model_KerasUBag_Level1_F02M03_clean.py',
                 'python Model_KerasUBag_Level1_F02M02_clean.py']

for i, nb in enumerate(notebook_list):
    print(100*'-')
    print('running notebook %d out of %d...' % (i+1, len(notebook_list)))
    print('notebook name: %s\n' % nb)
    os.system(nb)
    print(100*'-')
    print()

# run ensemble
print('running Ensemble...')
ensemble_notebook = 'python Model_ensemble.py'
os.system(ensemble_notebook)
