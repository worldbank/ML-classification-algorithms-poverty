import os
import sys
import process
import pandas as pd
from data.data import Data, DataConcat
from models import LGBM_model, CB_model, XGB_model

src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)


def predict(p_models={'xgboost': True,
                      'lightgbm': True,
                      'catboost': True}):
    filenames_dict = {
        'mwi': {'train': 'data/processed/mwi_aligned_hhold_train.csv',
              'test': 'data/processed/mwi_aligned_hhold_test.csv',
              'train_hh': '../../data/raw_mwi/mwi_aligned_hhold_train.csv',
              'test_hh': '../../data/raw_mwi/mwi_aligned_hhold_test.csv',
              'train_ind': 'data/processed/mwi_aligned_indiv_train.csv',
              'test_ind': 'data/processed/mwi_aligned_indiv_test.csv'
              },
        }

    data_mwi = Data()

    data_mwi.set_country('mwi')

    data_mwi.set_file_names(files_dict=filenames_dict['mwi'])

    data_mwi.load(load=True)
    # To reproduce the result in the final submission.
    # Saving data to a file changes this data due to rounding of numbers.
    
    data_dict = {'mwi': data_mwi}
    balances = {'mwi': False}

    # XGBoost prediction
    if p_models['xgboost']:
        params_XGB_mwi = {
                'learning_rate': 0.03,
                'max_depth': 3,
                'n_estimators': 1500,
                'silent': True,
                'objective': 'binary:logistic',
                'gamma': 0.3,
                'subsample': 0.7,
                'reg_alpha': 0.05
            }

        model_xgb_mwi = XGB_model(categ_conv=True)
        model_xgb_mwi.set_params(params=params_XGB_mwi)
        model_xgb_dict = {'mwi': model_xgb_mwi}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_XGB_dict = {'mwi': ['mwi_0', 'mwi_10', 'mwi_101', 'mwi_106', 'mwi_11', 'mwi_113', 'mwi_120', 'mwi_121', 'mwi_13', 'mwi_131', 'mwi_134', 'mwi_138', 'mwi_140', 'mwi_146', 'mwi_147', 'mwi_148', 'mwi_15', 'mwi_152', 'mwi_155', 'mwi_161', 'mwi_162', 'mwi_167', 'mwi_168', 'mwi_17', 'mwi_170', 'mwi_173', 'mwi_174', 'mwi_175', 'mwi_176', 'mwi_179', 'mwi_18', 'mwi_181', 'mwi_185', 'mwi_186', 'mwi_191', 'mwi_195', 'mwi_197', 'mwi_2', 'mwi_202', 'mwi_203', 'mwi_206', 'mwi_213', 'mwi_215', 'mwi_216', 'mwi_218', 'mwi_219', 'mwi_22', 'mwi_223', 'mwi_225', 'mwi_226', 'mwi_227', 'mwi_232', 'mwi_234', 'mwi_237', 'mwi_242', 'mwi_245', 'mwi_251', 'mwi_252', 'mwi_253', 'mwi_254', 'mwi_255', 'mwi_256', 'mwi_258', 'mwi_259', 'mwi_26', 'mwi_261', 'mwi_262', 'mwi_263', 'mwi_267', 'mwi_27', 'mwi_272', 'mwi_277', 'mwi_282', 'mwi_295', 'mwi_299', 'mwi_3', 'mwi_30', 'mwi_301', 'mwi_302', 'mwi_305', 'mwi_307', 'mwi_308', 'mwi_309', 'mwi_31', 'mwi_312', 'mwi_315', 'mwi_319', 'mwi_32', 'mwi_322', 'mwi_33', 'mwi_330', 'mwi_332', 'mwi_335', 'mwi_341', 'mwi_35', 'mwi_39', 'mwi_43', 'mwi_44', 'mwi_45', 'mwi_46', 'mwi_49', 'mwi_57', 'mwi_59', 'mwi_60', 'mwi_61', 'mwi_63', 'mwi_66', 'mwi_67', 'mwi_69', 'mwi_70', 'mwi_72', 'mwi_76', 'mwi_77', 'mwi_80', 'mwi_81', 'mwi_88', 'mwi_89', 'mwi_9', 'mwi_91', 'mwi_93', 'mwi_97', 'cat_n_mwi_25', 'cat_n_mwi_3', 'cat_n_mwi_36', 'cat_n_mwi_4', 'iid_cnt', 'mwi_105', 'mwi_114', 'mwi_229', 'cat_n_mwi_20', 'div_cat_iid_cat_n_mwi_25', 'mwi_14', 'mwi_6_1', 'cat_n_mwi_39']}

        process_xgb = process.processing(countries=['mwi'],
                                        balances=balances)
        process_xgb.set_data_dict(data_dict=data_dict)
        process_xgb.set_model_dict(model_dict=model_xgb_dict)
        # process_xgb.find_exclude()
        process_xgb.set_exclude_dict(exclude_XGB_dict)
        result_xgb = process_xgb.predict(model_name='xgboost', path='models/')

    # LightGBM prediction
    if p_models['lightgbm']:
        params_LGBM_mwi = {
                'learning_rate': 0.02,
                'max_depth': 6,
                'n_estimators': 942,
                'silent': True,
                'objective': 'binary',
                'subsample': 0.6,
                'reg_alpha': 0.02,
                'is_unbalance': True,
                'boosting_type': 'gbdt',
                'reg_lambda': 0.01,
                'random_state': 1
            }

        model_lgbm_mwi = LGBM_model(categ_conv=True)
        model_lgbm_mwi.set_params(params=params_LGBM_mwi)
        model_lgbm_dict = {'mwi': model_lgbm_mwi}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_LGBM_dict = {'mwi': ['mwi_0', 'mwi_10', 'mwi_101', 'mwi_105', 'mwi_106', 'mwi_11', 'mwi_112', 'mwi_113', 'mwi_115', 'mwi_120', 'mwi_121', 'mwi_13', 'mwi_131', 'mwi_134', 'mwi_138', 'mwi_141', 'mwi_15', 'mwi_152', 'mwi_155', 'mwi_161', 'mwi_162', 'mwi_167', 'mwi_168', 'mwi_170', 'mwi_173', 'mwi_174', 'mwi_175', 'mwi_176', 'mwi_18', 'mwi_181', 'mwi_185', 'mwi_191', 'mwi_195', 'mwi_197', 'mwi_202', 'mwi_203', 'mwi_206', 'mwi_215', 'mwi_216', 'mwi_218', 'mwi_219', 'mwi_223', 'mwi_225', 'mwi_232', 'mwi_237', 'mwi_242', 'mwi_245', 'mwi_251', 'mwi_252', 'mwi_253', 'mwi_254', 'mwi_255', 'mwi_256', 'mwi_258', 'mwi_259', 'mwi_26', 'mwi_261', 'mwi_262', 'mwi_263', 'mwi_267', 'mwi_27', 'mwi_272', 'mwi_275', 'mwi_282', 'mwi_292', 'mwi_295', 'mwi_299', 'mwi_3', 'mwi_30', 'mwi_301', 'mwi_307', 'mwi_308', 'mwi_309', 'mwi_31', 'mwi_312', 'mwi_319', 'mwi_32', 'mwi_322', 'mwi_33', 'mwi_330', 'mwi_332', 'mwi_335', 'mwi_338', 'mwi_341', 'mwi_35', 'mwi_39', 'mwi_43', 'mwi_44', 'mwi_46', 'mwi_47', 'mwi_49', 'mwi_57', 'mwi_59', 'mwi_60', 'mwi_63', 'mwi_66', 'mwi_67', 'mwi_69', 'mwi_70', 'mwi_72', 'mwi_77', 'mwi_80', 'mwi_81', 'mwi_88', 'mwi_89', 'mwi_9', 'mwi_91', 'mwi_93']}

        process_lgbm = process.processing(countries=['mwi'],
                                        balances=balances)
        process_lgbm.set_data_dict(data_dict=data_dict)
        process_lgbm.set_model_dict(model_dict=model_lgbm_dict)
        process_lgbm.set_exclude_dict(exclude_LGBM_dict)
        # process_lgbm.find_exclude()
        result_lgbm = process_lgbm.predict(model_name='lightgbm', path='models/')

    # Catboost prediction
    if p_models['catboost']:
        params_CB_mwi = {
                'iterations': 5000,
                'learning_rate': 0.03,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss',
                'random_seed': 1,
                'logging_level': 'Silent',
            }

        model_cb_mwi = CB_model(categ_conv=True)
        model_cb_mwi.set_params(params=params_CB_mwi)
        model_cb_dict = {'mwi': model_cb_mwi}

        # List of columns to delete obtained via find_exclude function and cross-validation
        exclude_CB_dict = {'mwi': ['mwi_0', 'mwi_10', 'mwi_106', 'mwi_113', 'mwi_114', 'mwi_115', 'mwi_120', 'mwi_138', 'mwi_15', 'mwi_173', 'mwi_174', 'mwi_175', 'mwi_181', 'mwi_185', 'mwi_191', 'mwi_195', 'mwi_202', 'mwi_206', 'mwi_215', 'mwi_216', 'mwi_218', 'mwi_223', 'mwi_245', 'mwi_250', 'mwi_251', 'mwi_252', 'mwi_253', 'mwi_254', 'mwi_255', 'mwi_256', 'mwi_263', 'mwi_272', 'mwi_277', 'mwi_295', 'mwi_299', 'mwi_308', 'mwi_309', 'mwi_32', 'mwi_33', 'mwi_330', 'mwi_39', 'mwi_43', 'mwi_44', 'mwi_57', 'mwi_59', 'mwi_63', 'mwi_69', 'mwi_6_1', 'mwi_70', 'mwi_72', 'mwi_77', 'mwi_81', 'mwi_88', 'mwi_89', 'mwi_93', 'cat_n_mwi_10', 'cat_n_mwi_15', 'cat_n_mwi_20', 'cat_n_mwi_22', 'cat_n_mwi_25', 'cat_n_mwi_33', 'cat_n_mwi_35', 'cat_n_mwi_39', 'cat_n_mwi_4', 'cat_n_mwi_5', 'cat_n_mwi_6', 'cat_n_mwi_8', 'cat_n_mwi_9', 'mwi_101', 'mwi_11', 'mwi_147', 'mwi_155', 'mwi_170', 'mwi_18', 'mwi_203', 'mwi_338', 'mwi_35', 'mwi_49', 'mwi_67', 'cat_n_mwi_11', 'cat_n_mwi_21', 'cat_n_mwi_30', 'cat_n_mwi_37', 'div_cat_iid_cat_n_mwi_16', 'mwi_105', 'mwi_14', 'mwi_149', 'mwi_197', 'mwi_26', 'mwi_261', 'mwi_302', 'mwi_312', 'mwi_319', 'mwi_328', 'mwi_32_1', 'mwi_341', 'mwi_9', 'cat_n_mwi_28', 'div_cat_iid_cat_n_mwi_32', 'mwi_121', 'mwi_125', 'mwi_131', 'mwi_161', 'mwi_17', 'mwi_192', 'mwi_229', 'mwi_259', 'mwi_60', 'mwi_80', 'cat_n_mwi_1', 'cat_n_mwi_29', 'mwi_13', 'mwi_134', 'mwi_176', 'mwi_182', 'mwi_213', 'mwi_22', 'mwi_267', 'mwi_301', 'mwi_31', 'mwi_146', 'mwi_162', 'mwi_27', 'mwi_152', 'mwi_189', 'mwi_292', 'mwi_3', 'mwi_65']}

        process_cb = process.processing(countries=['mwi'],
                                        balances=balances)
        process_cb.set_data_dict(data_dict=data_dict)
        process_cb.set_model_dict(model_dict=model_cb_dict)
        # process_cb.find_exclude()
        process_cb.set_exclude_dict(exclude_CB_dict)
        result_cb = process_cb.predict(model_name='catboost', path='models/')

    # Create submission
    submission = pd.DataFrame(index=result_cb.index)
    submission['country'] = result_cb.country
    submission['poor'] = (result_xgb.poor * 0.4 +
                          result_cb.poor * 0.4 +
                          result_lgbm.poor * 0.2)

    process_cb.save_csv(submission, clf_model_name='combine', path='models/')


if __name__ == '__main__':
    predict()
