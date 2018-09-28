# -*- coding: utf-8 -*-
import os
import click
import logging
import data
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_hh_train_filepath', type=click.Path())
@click.argument('input_ind_train_filepath', type=click.Path())
@click.argument('input_hh_test_filepath', type=click.Path())
@click.argument('input_ind_test_filepath', type=click.Path())
@click.argument('output_train_filepath', type=click.Path())
@click.argument('output_test_filepath', type=click.Path())
@click.argument('output_train_ind_filepath', type=click.Path())
@click.argument('output_test_ind_filepath', type=click.Path())
def main(input_hh_train_filepath,
         input_ind_train_filepath,
         input_hh_test_filepath,
         input_ind_test_filepath,
         output_train_filepath,
         output_test_filepath,
         output_train_ind_filepath,
         output_test_ind_filepath
         ):
    """ Runs data processing scripts to turn raw data from (../../data/raw_mwi/) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    country = input_hh_train_filepath.split('_')[0]
    data_individual = data.DataInd()
    files_dict = {'train': '../../data/raw_mwi/{0}'.format(input_ind_train_filepath),
                  'test': '../../data/raw_mwi/{0}'.format(input_ind_test_filepath)}
    data_individual.set_country(country)
    data_individual.set_file_names(files_dict=files_dict)
    data_individual.load(load=False, cat_enc=False)
    files_dict = {'train': 'data/processed/{0}'.format(output_train_ind_filepath),
                  'test': 'data/processed/{0}'.format(output_test_ind_filepath)}
    data_individual.save(files_dict=files_dict, poor=False)

    files_dict = {'train_hh': '../../data/raw_mwi/{0}'.format(input_hh_train_filepath),
                  'test_hh': '../../data/raw_mwi/{0}'.format(input_hh_test_filepath),
                  'train_ind': 'data/processed/{0}'.format(output_train_ind_filepath),
                  'test_ind': 'data/processed/{0}'.format(output_test_ind_filepath)}
    data_concat = data.DataConcat()
    data_concat.set_file_names(files_dict=files_dict)
    data_concat.set_country(country)
    data_concat.load(load=False, cat_enc=False)
    files_dict = {'train': 'data/processed/{0}'.format(output_train_filepath),
                  'test': 'data/processed/{0}'.format(output_test_filepath)}
    data_concat.save(files_dict=files_dict)

    files_dict = {'train': '../../data/raw_mwi/{0}'.format(input_hh_train_filepath),
                  'test': '../../data/raw_mwi/{0}'.format(input_hh_test_filepath)}
    data_simple = data.Data()
    data_simple.set_file_names(files_dict=files_dict)
    data_simple.set_country(country)
    data_simple.load(load=False)
    files_dict = {'train': 'data/processed/{0}'.format(input_hh_train_filepath),
                  'test': 'data/processed/{0}'.format(input_hh_test_filepath)}
    data_simple.save(files_dict=files_dict)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
