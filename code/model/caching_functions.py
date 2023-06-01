import logging
import pickle
import os
from typing import List

from code.paths import TMP_FOLDER


def generate_cache_parameters_from_command_line_argument(arguments: list[str]):
    # Generate cache parameters
    cache_parameters: dict[str, bool] = {
        "transport_network": False,
        "agents": False,
        "sc_network": False,
        "logistic_routes": False
    }
    if len(arguments) > 2:
        accepted_script_arguments: list[str] = [
            'same_transport_network_new_agents',
            'same_agents_new_sc_network',
            'same_sc_network_new_logistic_routes',
            'same_logistic_routes',
        ]
        argument = arguments[2]
        if argument not in accepted_script_arguments:
            raise ValueError(f"Argument {argument} is not valid.\
                Possible values are: " + ','.join(accepted_script_arguments))
        cache_parameters: dict[str, bool] = {
            "transport_network": False,
            "agents": False,
            "sc_network": False,
            "logistic_routes": False
        }
        if argument == accepted_script_arguments[0]:
            cache_parameters['transport_network'] = True
        if argument == accepted_script_arguments[1]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
        if argument == accepted_script_arguments[2]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = True
        if argument == accepted_script_arguments[3]:
            cache_parameters['transport_network'] = True
            cache_parameters['agents'] = True
            cache_parameters['sc_network'] = True
            cache_parameters['logistic_routes'] = True
    return cache_parameters


def cache_agent_data(data_dic):
    pickle_filename = TMP_FOLDER / 'firms_households_countries_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Firms, households, and countries saved in tmp folder: {pickle_filename}')


def cache_transport_network(data_dic):
    pickle_filename = TMP_FOLDER / 'transport_network_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Transport network saved in tmp folder: {pickle_filename}')


def cache_sc_network(data_dic):
    pickle_filename = TMP_FOLDER / 'supply_chain_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Supply chain saved in tmp folder: {pickle_filename}')


def cache_logistic_routes(data_dic):
    pickle_filename = TMP_FOLDER / 'logistic_routes_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Logistics routes saved in tmp folder: {pickle_filename}')


def load_cached_agent_data():
    pickle_filename = TMP_FOLDER / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sector_table = tmp_data['sector_table']
    # loaded_present_sectors = tmp_data['present_sectors']
    # loaded_flow_types_to_export = tmp_data['flow_types_to_export']
    loaded_firm_table = tmp_data['firm_table']
    loaded_household_table = tmp_data['household_table']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Firms, households, and countries generated from temp file.')
    logging.info("Nb firms: " + str(len(loaded_firm_list)))
    logging.info("Nb households: " + str(len(loaded_household_list)))
    logging.info("Nb countries: " + str(len(loaded_country_list)))
    return loaded_sector_table, loaded_firm_list, loaded_firm_table, \
        loaded_household_list, loaded_household_table, loaded_country_list


def load_cached_transaction_table():
    pickle_filename = TMP_FOLDER / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transaction_table = tmp_data['transaction_table']
    return loaded_transaction_table


def load_cached_transport_network():
    pickle_filename = TMP_FOLDER / 'transport_network_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transport_network = tmp_data['transport_network']
    loaded_transport_nodes = tmp_data['transport_nodes']
    loaded_transport_edges = tmp_data['transport_edges']
    logging.info('Transport network generated from temp file.')
    return loaded_transport_network, loaded_transport_nodes, loaded_transport_edges


def load_cached_sc_network():
    pickle_filename = TMP_FOLDER / 'supply_chain_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sc_network = tmp_data['supply_chain_network']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Supply chain generated from temp file.')
    return loaded_sc_network, loaded_firm_list, loaded_household_list, loaded_country_list


def load_cached_logistic_routes():
    pickle_filename = TMP_FOLDER / 'logistic_routes_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sc_network = tmp_data['supply_chain_network']
    loaded_transport_network = tmp_data['transport_network']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Logistic routes generated from temp file.')
    return loaded_sc_network, loaded_transport_network, loaded_firm_list, loaded_household_list, loaded_country_list
