# Import modules
import sys
import time
import yaml
from datetime import datetime
import importlib
import pickle

# Import functions and classes
from builder import *
from simulations import *
from export_functions import *
from class_observer import Observer
from paths import ROOT_FOLDER

# Import parameters. It should be in this specific order.
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, project_path)
sys.path.insert(1, ROOT_FOLDER)

from parameter.parameters import *
from parameter.filepaths import *
from code.disruption.disruption import Disruption, DisruptionList

# Check that the script is called correctly
accepted_script_arguments = [
    'same_transport_network_new_agents',
    'same_agents_new_sc_network',
    'same_sc_network_new_logistic_routes',
    'same_logistic_routes',
]
if len(sys.argv) > 2:
    raise ValueError('The script does not take more than 1 arguments')
if len(sys.argv) > 1:
    if sys.argv[1] not in accepted_script_arguments:
        raise ValueError("First argument " + sys.argv[1] + " is not valid.\
            Possible values are: " + ','.join(accepted_script_arguments))

# Start run
t0 = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


def extract_final_list_of_sector(firm_list: list):
    n = len(firm_list)
    present_sectors = list(set([firm.sector for firm in firm_list]))
    present_sectors.sort()
    flow_types_to_export = present_sectors + ['domestic_B2C', 'domestic_B2B', 'transit', 'import', 'export', 'total']
    logging.info('Firm_list created, size is: ' + str(n))
    logging.info('Sectors present are: ' + str(present_sectors))
    return n, present_sectors, flow_types_to_export


def create_export_folder(export: dict, input_folder: str, timestamp: str) -> str:
    if any(list(export.values())):
        exp_folder = ROOT_FOLDER / input_folder / timestamp
        if not os.path.isdir(ROOT_FOLDER / input_folder):
            os.mkdir(ROOT_FOLDER / input_folder)
        os.mkdir(exp_folder)
        exportParameters(exp_folder)

    else:
        exp_folder = None

    return exp_folder


def adjust_logging_behavior(export: dict, exp_folder: str, logging_level):
    if export['log']:
        importlib.reload(logging)
        logging.basicConfig(
            filename=os.path.join(exp_folder, 'exp.log'),
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        importlib.reload(logging)
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def cache_agent_data(data_dic):
    pickle_filename = ROOT_FOLDER / 'tmp' / 'firms_households_countries_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Firms, households, and countries saved in tmp folder: {pickle_filename}')


def cache_transport_network(data_dic):
    pickle_filename = ROOT_FOLDER / 'tmp' / 'transport_network_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Transport network saved in tmp folder: {pickle_filename}')


def cache_sc_network(data_dic):
    pickle_filename = ROOT_FOLDER / 'tmp' / 'supply_chain_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Supply chain saved in tmp folder: {pickle_filename}')


def cache_logistic_routes(data_dic):
    pickle_filename = ROOT_FOLDER / 'tmp' / 'logistic_routes_pickle'
    pickle.dump(data_dic, open(pickle_filename, 'wb'))
    logging.info(f'Logistics routes saved in tmp folder: {pickle_filename}')


def load_cached_agent_data():
    pickle_filename = ROOT_FOLDER / 'tmp' / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sector_table = tmp_data['sector_table']
    loaded_present_sectors = tmp_data['present_sectors']
    loaded_flow_types_to_export = tmp_data['flow_types_to_export']
    loaded_firm_table = tmp_data['firm_table']
    loaded_household_table = tmp_data['household_table']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Firms, households, and countries generated from temp file.')
    logging.info("Nb firms: " + str(len(loaded_firm_list)))
    logging.info("Nb households: " + str(len(loaded_household_list)))
    logging.info("Nb countries: " + str(len(loaded_country_list)))
    return loaded_sector_table, loaded_firm_table, loaded_present_sectors, loaded_flow_types_to_export, \
        loaded_firm_list, loaded_household_table, loaded_household_list, loaded_country_list


def load_cached_transaction_table():
    pickle_filename = ROOT_FOLDER / 'tmp' / 'firms_households_countries_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transaction_table = tmp_data['transaction_table']
    return loaded_transaction_table


def load_cached_transport_network():
    pickle_filename = ROOT_FOLDER / 'tmp' / 'transport_network_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_transport_network = tmp_data['transport_network']
    loaded_transport_edges = tmp_data['transport_edges']
    loaded_transport_nodes = tmp_data['transport_nodes']
    logging.info('Transport network generated from temp file.')
    return loaded_transport_network, loaded_transport_edges, loaded_transport_nodes


def load_cached_sc_network():
    pickle_filename = ROOT_FOLDER / 'tmp' / 'supply_chain_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sc_network = tmp_data['supply_chain_network']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Supply chain generated from temp file.')
    return loaded_sc_network, loaded_firm_list, loaded_household_list, loaded_country_list


def load_cached_logistic_routes():
    pickle_filename = ROOT_FOLDER / 'tmp' / 'logistic_routes_pickle'
    tmp_data = pickle.load(open(pickle_filename, 'rb'))
    loaded_sc_network = tmp_data['supply_chain_network']
    loaded_transport_network = tmp_data['transport_network']
    loaded_firm_list = tmp_data['firm_list']
    loaded_household_list = tmp_data['household_list']
    loaded_country_list = tmp_data['country_list']
    logging.info('Logistic routes generated from temp file.')
    return loaded_sc_network, loaded_transport_network, loaded_firm_list, loaded_household_list, loaded_country_list


# If there is sth to export, then we create the output folder
exp_folder = create_export_folder(export, input_folder, timestamp)

# Set logging parameters
adjust_logging_behavior(export, exp_folder, logging_level)

logging.info('Simulation ' + timestamp + ' starting using ' + input_folder + ' input data.')

# Create transport network
with open(filepaths['transport_parameters'], "r") as yamlfile:
    transport_params = yaml.load(yamlfile, Loader=yaml.FullLoader)

# Creating the transport network consumes time
# To accelerate, we enable storing the transport network as pickle for later reuse
if len(sys.argv) < 2:
    T, transport_nodes, transport_edges = createTransportNetwork(transport_modes, filepaths, transport_params)
    data_to_cache = {
        "transport_network": T,
        'transport_edges': transport_edges,
        'transport_nodes': transport_nodes
    }
    cache_transport_network(data_to_cache)
elif sys.argv[1] in accepted_script_arguments:
    T, transport_edges, transport_nodes = load_cached_transport_network()
else:
    raise ValueError('Argument error')
# Generate weight
logging.info('Generating shortest-path weights on transport network')
T.defineWeights(route_optimization_weight, logistics_modes)
T.log_km_per_transport_modes()  # Print data on km per modes

# Create firms, households, and countries
if (len(sys.argv) < 2) or (sys.argv[1] == "same_transport_network_new_agents"):
    # Filter sectors
    logging.info('Filtering the sectors based on their output. ' +
                 "Cutoff type is " + cutoff_sector_output['type'] +
                 ", cutoff value is " + str(cutoff_sector_output['value']))
    sector_table = pd.read_csv(filepaths['sector_table'])
    filtered_sectors = filterSector(sector_table,
                                    cutoff_sector_output=cutoff_sector_output,
                                    cutoff_sector_demand=cutoff_sector_demand,
                                    combine_sector_cutoff=combine_sector_cutoff,
                                    sectors_to_include=sectors_to_include,
                                    sectors_to_exclude=sectors_to_exclude)
    output_selected = sector_table.loc[sector_table['sector'].isin(filtered_sectors), 'output'].sum()
    final_demand_selected = sector_table.loc[sector_table['sector'].isin(filtered_sectors), 'final_demand'].sum()
    logging.info(
        str(len(filtered_sectors)) + ' sectors selected over ' + str(sector_table.shape[0]) + ' representing ' +
        "{:.0f}%".format(output_selected / sector_table['output'].sum() * 100) + ' of total output and ' +
        "{:.0f}%".format(final_demand_selected / sector_table['final_demand'].sum() * 100) + ' of final demand'
    )
    logging.info('The filtered sectors are: ' + str(filtered_sectors))

    logging.info('Generating the firms')
    if firm_data_type == "disaggregating IO":
        firm_table, firm_table_per_adminunit = defineFirmsFromGranularEcoData(
            filepath_adminunit_economic_data=filepaths['adminunit_data'],
            sectors_to_include=filtered_sectors,
            transport_nodes=transport_nodes,
            filepath_sector_table=filepaths['sector_table']
        )
        nb_firms = 'all'
        logging.info('Creating firm_list. nb_firms: ' + str(nb_firms) +
                     ' reactivity_rate: ' + str(reactivity_rate) +
                     ' utilization_rate: ' + str(utilization_rate))
        firm_list = createFirms(
            firm_table=firm_table,
            keep_top_n_firms=nb_firms,
            reactivity_rate=reactivity_rate,
            utilization_rate=utilization_rate
        )
    elif firm_data_type == "supplier-buyer network":
        firm_table = defineFirmsFromNetworkData(
            filepath_firm_table=filepaths['firm_table'],
            filepath_location_table=filepaths['location_table'],
            sectors_to_include=sectors_to_include,
            transport_nodes=transport_nodes,
            filepath_sector_table=filepaths['sector_table'])
        nb_firms = 'all'
        logging.info('Creating firm_list. nb_firms: ' + str(nb_firms) +
                     ' reactivity_rate: ' + str(reactivity_rate) +
                     ' utilization_rate: ' + str(utilization_rate))
        firm_list = createFirms(
            firm_table=firm_table,
            keep_top_n_firms=nb_firms,
            reactivity_rate=reactivity_rate,
            utilization_rate=utilization_rate
        )
    else:
        raise ValueError(firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'")

    n, present_sectors, flow_types_to_export = extract_final_list_of_sector(firm_list)

    # Create households
    logging.info('Defining the number of households to generate and their purchase plan')
    household_table, household_sector_consumption = defineHouseholds(
        sector_table=sector_table,
        filepath_adminunit_data=filepaths['adminunit_data'],
        filtered_sectors=present_sectors,
        pop_cutoff=pop_cutoff,
        pop_density_cutoff=pop_density_cutoff,
        local_demand_cutoff=local_demand_cutoff,
        transport_nodes=transport_nodes,
        time_resolution=time_resolution,
        target_units=monetary_units_in_model,
        input_units=monetary_units_inputed
    )
    cond_no_household = ~firm_table['odpoint'].isin(household_table['odpoint'])
    if cond_no_household.sum() > 0:
        logging.info('We add local households for firms')
        household_table, household_sector_consumption = addHouseholdsForFirms(
            firm_table=firm_table,
            household_table=household_table,
            filepath_adminunit_data=filepaths['adminunit_data'],
            sector_table=sector_table,
            filtered_sectors=present_sectors,
            time_resolution=time_resolution,
            target_units=monetary_units_in_model,
            input_units=monetary_units_inputed
        )
    household_list = createHouseholds(
        household_table=household_table,
        household_sector_consumption=household_sector_consumption
    )

    # Loading the technical coefficients
    if firm_data_type == "disaggregating IO":
        import_code = sector_table.loc[sector_table['type'] == 'imports', 'sector'].iloc[0]  # usually it is IMP
        firm_list = loadTechnicalCoefficients(firm_list, filepaths['tech_coef'], io_cutoff, import_code)

    elif firm_data_type == "supplier-buyer network":
        firm_table, transaction_table = calibrateInputMix(
            firm_list=firm_list,
            firm_table=firm_table,
            sector_table=sector_table,
            filepath_transaction_table=filepaths['transaction_table']
        )

    else:
        raise ValueError(firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'")

    # Loading the inventories
    firm_list = loadInventories(
        firm_list=firm_list,
        inventory_duration_target=inventory_duration_target,
        filepath_inventory_duration_targets=filepaths['inventory_duration_targets'],
        extra_inventory_target=extra_inventory_target,
        inputs_with_extra_inventories=inputs_with_extra_inventories,
        buying_sectors_with_extra_inventories=buying_sectors_with_extra_inventories,
        min_inventory=1
    )

    # Create agents: Countries
    country_list = createCountries(
        filepath_imports=filepaths['imports'],
        filepath_exports=filepaths['exports'],
        filepath_transit_matrix=filepaths['transit_matrix'],
        transport_nodes=transport_nodes,
        present_sectors=present_sectors,
        countries_to_include=countries_to_include,
        time_resolution=time_resolution,
        target_units=monetary_units_in_model,
        input_units=monetary_units_inputed
    )

    ### Specify the weight of a unit worth of good, which may differ according to sector, or even to each firm/countries
    # Note that for imports, i.e. for the goods delivered by a country, and for transit flows, we do not disentangle sectors
    # In this case, we use an average.
    firm_list, country_list, sector_to_usdPerTon = loadTonUsdEquivalence(
        sector_table=sector_table,
        firm_list=firm_list,
        country_list=country_list
    )

    # Save to tmp folder
    data_to_cache = {
        "sector_table": sector_table,
        'firm_table': firm_table,
        'present_sectors': present_sectors,
        'flow_types_to_export': flow_types_to_export,
        'firm_list': firm_list,
        'household_table': household_table,
        'household_list': household_list,
        'country_list': country_list
    }
    if firm_data_type == "supplier-buyer network":
        data_to_cache['transaction_table'] = transaction_table
    cache_agent_data(data_to_cache)


elif sys.argv[1] in ["same_agents_new_sc_network", "same_sc_network_new_logistic_routes", "same_logistic_routes"]:
    sector_table, firm_table, present_sectors, flow_types_to_export, \
        firm_list, household_table, household_list, country_list = load_cached_agent_data()
    if firm_data_type == "supplier-buyer network":
        transaction_table = load_cached_transaction_table()

else:
    raise ValueError('Argument error')

# Locate firms and households on transport network
T.locate_firms_on_nodes(firm_list, transport_nodes)
T.locate_households_on_nodes(household_list, transport_nodes)
logging.info('Firms and household located on the transport network')
# Export transport network
if export['transport']:
    transport_nodes.to_file(os.path.join(exp_folder, "transport_nodes.geojson"), driver='GeoJSON')
    transport_edges.to_file(os.path.join(exp_folder, "transport_edges.geojson"), driver='GeoJSON')

# Create supply chain network
if (len(sys.argv) < 2) or (sys.argv[1] in ["same_transport_network_new_agents", "same_agents_new_sc_network"]):
    logging.info('The supply chain graph is being created. nb_suppliers_per_input: ' + str(nb_suppliers_per_input))
    G = nx.DiGraph()

    logging.info('Households are selecting their retailers (domestic B2C flows)')
    for household in household_list:
        household.select_suppliers(G, firm_list, firm_table, nb_suppliers_per_input, weight_localization_household)

    logging.info('Exporters are being selected by purchasing countries (export B2B flows)')
    logging.info('and trading countries are being connected (transit flows)')
    for country in country_list:
        country.select_suppliers(G, firm_list, country_list, sector_table, transport_nodes)

    logging.info(
        'Firms are selecting their domestic and international suppliers (import B2B flows) (domestic B2B flows).' +
        ' Weight localisation is ' + str(weight_localization_firm))
    import_code = sector_table.loc[sector_table['type'] == 'imports', 'sector'].iloc[0]

    if firm_data_type == "disaggregating IO":
        for firm in firm_list:
            firm.select_suppliers(G, firm_list, country_list, nb_suppliers_per_input, weight_localization_firm,
                                  import_code=import_code)

    elif firm_data_type == "supplier-buyer network":
        for firm in firm_list:
            inputed_supplier_links = transaction_table[transaction_table['buyer_id'] == firm.pid]
            output = firm_table.set_index('id').loc[firm.pid, "output"]
            firm.select_suppliers_from_data(G, firm_list, country_list, inputed_supplier_links, output,
                                            import_code=import_code)

    else:
        raise ValueError(firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'")

    logging.info('The nodes and edges of the supplier--buyer have been created')
    if export['sc_network_summary']:
        exportSupplyChainNetworkSummary(G, firm_list, exp_folder)

    # Save to tmp folder
    data_to_cache = {
        "supply_chain_network": G,
        'firm_list': firm_list,
        'household_list': household_list,
        'country_list': country_list
    }
    cache_sc_network(data_to_cache)

elif sys.argv[1] in ["same_sc_network_new_logistic_routes", "same_logistic_routes"]:
    G, firm_list, household_list, country_list = load_cached_sc_network()

else:
    raise ValueError('Argument error')

logging.info('Compute the orders on each supplier--buyer link')
setInitialSCConditions(T, G, firm_list,
                       country_list, household_list, initialization_mode="equilibrium")

# Coupling transportation network T and production network G
logging.info('The supplier--buyer graph is being connected to the transport network')
if (len(sys.argv) < 2) or (sys.argv[1] in ["same_transport_network_new_agents", "same_agents_new_sc_network",
                                           "same_sc_network_new_logistic_routes"]):
    logging.info('Each B2B and transit edge is being linked to a route of the transport network')
    transport_modes = pd.read_csv(filepaths['transport_modes'])
    logging.info(
        'Routes for transit and import flows are being selected by trading countries')
    for country in country_list:
        country.choose_initial_routes(G, T, transport_modes, account_capacity, monetary_units_in_model)
    logging.info(
        'Routes for exports and B2B domestic flows are being selected by domestic firms')
    for firm in firm_list:
        if firm.sector_type not in sectors_no_transport_network:
            firm.choose_initial_routes(G, T, transport_modes, account_capacity, monetary_units_in_model)
    # Save to tmp folder
    data_to_cache = {
        'transport_network': T,
        "supply_chain_network": G,
        'firm_list': firm_list,
        'household_list': household_list,
        'country_list': country_list
    }
    cache_logistic_routes(data_to_cache)

elif sys.argv[1] == "same_logistic_routes":
    G, T, firm_list, household_list, country_list = load_cached_logistic_routes()

else:
    raise ValueError('Argument error')

logging.info('The supplier--buyer graph is now connected to the transport network')

logging.info("Initialization completed, " + str((time.time() - t0) / 60) + " min")

######################################################
######################################################
######################################################
if disruption_analysis is None:
    logging.info("No disruption. Simulation of the initial state")
    t0 = time.time()

    # comments: not sure if the other initialization mode is (i) working and (ii) useful
    logging.info("Calculating the equilibrium")
    setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list,
                           country_list=country_list, household_list=household_list, initialization_mode="equilibrium")

    obs = Observer(
        firm_list=firm_list,
        Tfinal=0,
        specific_edges_to_monitor=specific_edges_to_monitor
    )

    # if export['district_sector_table']:
    #     exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

    if export['firm_table'] or export['odpoint_table']:
        exportFirmODPointTable(firm_list, firm_table, household_table, filepaths['roads_nodes'],
                               export_firm_table=export['firm_table'], export_odpoint_table=export['odpoint_table'],
                               export_folder=exp_folder)

    if export['country_table']:
        exportCountryTable(country_list, export_folder=exp_folder)

    if export['edgelist_table']:
        exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

    if export['inventories']:
        exportInventories(firm_list, export_folder=exp_folder)

    ### Run the simulation at the initial state
    logging.info("Simulating the initial state")
    runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list,
                   country_list=country_list, household_list=household_list,
                   disruptions=None,
                   congestion=congestion,
                   route_optimization_weight=route_optimization_weight,
                   logistics_modes=logistics_modes,
                   explicit_service_firm=explicit_service_firm,
                   sectors_no_transport_network=sectors_no_transport_network,
                   propagate_input_price_change=propagate_input_price_change,
                   rationing_mode=rationing_mode,
                   observer=obs,
                   time_step=0,
                   export_folder=exp_folder,
                   export_flows=export['flows'],
                   flow_types_to_export=flow_types_to_export,
                   transport_edges=transport_edges,
                   export_sc_flow_analysis=export['sc_flow_analysis'],
                   monetary_units_in_model=monetary_units_in_model,
                   cost_repercussion_mode=cost_repercussion_mode)

    if export['agent_data']:
        exportAgentData(obs, exp_folder)

    logging.info("Simulation completed, " + str((time.time() - t0) / 60) + " min")




######################################################
######################################################
######################################################
elif disruption_analysis['type'] == "compound":
    logging.info('Compound events simulation')
    t0 = time.time()

    disruption_list = DisruptionList.from_disruption_description(
        disruption_analysis, transport_edges, firm_table
    )

    logging.info(f"{len(disruption_list)} disruption(s) will occur")
    disruption_list.print_info()

    logging.info("Calculating the equilibrium")
    setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list,
                           country_list=country_list, household_list=household_list, initialization_mode="equilibrium")

    Tfinal = duration_dic[disruption_list.end_time]
    obs = Observer(
        firm_list=firm_list,
        Tfinal=Tfinal,
        specific_edges_to_monitor=specific_edges_to_monitor
    )
    obs.disruption_time = 1  # time of first disruption

    logging.info("Simulating the initial state")
    runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list,
                   country_list=country_list, household_list=household_list,
                   disruptions=None,
                   congestion=congestion,
                   route_optimization_weight=route_optimization_weight,
                   logistics_modes=logistics_modes,
                   explicit_service_firm=explicit_service_firm,
                   sectors_no_transport_network=sectors_no_transport_network,
                   propagate_input_price_change=propagate_input_price_change,
                   rationing_mode=rationing_mode,
                   observer=obs,
                   time_step=0,
                   export_folder=exp_folder,
                   export_flows=export['flows'],
                   flow_types_to_export=flow_types_to_export,
                   transport_edges=transport_edges,
                   export_sc_flow_analysis=export['sc_flow_analysis'],
                   monetary_units_in_model=monetary_units_in_model,
                   cost_repercussion_mode=cost_repercussion_mode)

    logging.info("Do initial exports")
    if export['firm_table'] or export['odpoint_table']:
        exportFirmODPointTable(firm_list, firm_table, household_table, filepaths['roads_nodes'],
                               export_firm_table=export['firm_table'], export_odpoint_table=export['odpoint_table'],
                               export_folder=exp_folder)

    if export['country_table']:
        exportCountryTable(country_list, export_folder=exp_folder)

    if export['edgelist_table']:
        exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

    if export['inventories']:
        exportInventories(firm_list, export_folder=exp_folder)

    if export['impact_per_firm']:
        extra_spending_export_file, missing_consumption_export_file = \
            initializeResPerFirmExportFile(exp_folder, firm_list)

    logging.info(str(len(disruption_analysis['events'])) + ' disruption events will occur.')
    logging.info('Simulation will last at max ' + str(Tfinal) + ' time steps.')

    logging.info("Starting time loop")
    for t in range(1, Tfinal + 1):
        logging.info('Time t=' + str(t))
        runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list,
                       country_list=country_list, household_list=household_list,
                       disruptions=disruption_list,
                       congestion=congestion,
                       route_optimization_weight=route_optimization_weight,
                       logistics_modes=logistics_modes,
                       explicit_service_firm=explicit_service_firm,
                       sectors_no_transport_network=sectors_no_transport_network,
                       propagate_input_price_change=propagate_input_price_change,
                       rationing_mode=rationing_mode,
                       observer=obs,
                       time_step=t,
                       export_folder=exp_folder,
                       export_flows=export['flows'],
                       flow_types_to_export=flow_types_to_export,
                       transport_edges=transport_edges,
                       export_sc_flow_analysis=False,
                       monetary_units_in_model=monetary_units_in_model,
                       cost_repercussion_mode=cost_repercussion_mode)
        logging.debug('End of t=' + str(t))

        if (t > max([disruption.start_time for disruption in disruption_list])) and epsilon_stop_condition:
            household_extra_spending = sum([household.extra_spending for household in household_list])
            household_consumption_loss = sum([household.consumption_loss for household in household_list])
            country_extra_spending = sum([country.extra_spending for country in country_list])
            country_consumption_loss = sum([country.consumption_loss for country in country_list])
            if (household_extra_spending <= epsilon_stop_condition) & \
                    (household_consumption_loss <= epsilon_stop_condition) & \
                    (country_extra_spending <= epsilon_stop_condition) & \
                    (country_consumption_loss <= epsilon_stop_condition):
                logging.info('Household and country extra spending and consumption loss are at pre-disruption value. ' \
                             + "Simulation stops.")
                break

    computation_time = time.time() - t0
    logging.info("Time loop completed, {:.02f} min".format(computation_time / 60))

    disrupted_nodes = disruption_list.filter_type('transport_node').get_id_list()
    # disrupted_nodes = [item for sublist in disrupted_nodes for item in sublist]
    obs.evaluate_results(T, household_list, disrupted_nodes,
                         epsilon_stop_condition, per_firm=export['impact_per_firm'])

    if export['time_series']:
        exportTimeSeries(obs, exp_folder)

    if export['impact_per_firm']:
        writeResPerFirmResults(extra_spending_export_file,
                               missing_consumption_export_file, obs, "compound")

    # if export['agent_data']:
    #     exportAgentData(obs, export_folder=exp_folder)

    del obs

    exit()


######################################################
######################################################
######################################################
elif disruption_analysis['type'] == 'criticality':
    logging.info("Criticality analysis. Defining the list of disruptions")
    disruption_list = defineDisruptionList(disruption_analysis, transport_network=T,
                                           nodes=transport_nodes, edges=transport_edges,
                                           nodeedge_tested_topn=nodeedge_tested_topn,
                                           nodeedge_tested_skipn=nodeedge_tested_skipn)
    logging.info(str(len(disruption_list)) + " disruptions to simulates.")

    if export['criticality']:
        criticality_export_file = initializeCriticalityExportFile(export_folder=exp_folder)

    if export['impact_per_firm']:
        extra_spending_export_file, missing_consumption_export_file = \
            initializeResPerFirmExportFile(exp_folder, firm_list)

    ### Disruption Loop
    for disruption in disruption_list:
        logging.info(' ')
        logging.info("Simulating disruption " + str(disruption))
        t0 = time.time()

        ### Set initial conditions and create observer
        logging.info("Calculating the equilibrium")
        setInitialSCConditions(transport_network=T, sc_network=G, firm_list=firm_list,
                               country_list=country_list, household_list=household_list,
                               initialization_mode="equilibrium")

        Tfinal = duration_dic[disruption['duration']]
        obs = Observer(
            firm_list=firm_list,
            Tfinal=Tfinal,
            specific_edges_to_monitor=specific_edges_to_monitor
        )

        if disruption == disruption_list[0]:
            export_flows = export['flows']
        else:
            export_flows = False

        logging.info("Simulating the initial state")
        runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list,
                       country_list=country_list, household_list=household_list,
                       disruptions=None,
                       congestion=congestion,
                       route_optimization_weight=route_optimization_weight,
                       logistics_modes=logistics_modes,
                       explicit_service_firm=explicit_service_firm,
                       sectors_no_transport_network=sectors_no_transport_network,
                       propagate_input_price_change=propagate_input_price_change,
                       rationing_mode=rationing_mode,
                       observer=obs,
                       time_step=0,
                       export_folder=exp_folder,
                       export_flows=export_flows,
                       flow_types_to_export=flow_types_to_export,
                       transport_edges=transport_edges,
                       export_sc_flow_analysis=export['sc_flow_analysis'],
                       monetary_units_in_model=monetary_units_in_model,
                       cost_repercussion_mode=cost_repercussion_mode)

        if disruption == disruption_list[0]:
            # if export['district_sector_table']:
            #     exportDistrictSectorTable(filtered_district_sector_table, export_folder=exp_folder)

            if export['firm_table'] or export['odpoint_table']:
                exportFirmODPointTable(firm_list, firm_table, household_table, filepaths['roads_nodes'],
                                       export_firm_table=export['firm_table'],
                                       export_odpoint_table=export['odpoint_table'],
                                       export_folder=exp_folder)

            if export['country_table']:
                exportCountryTable(country_list, export_folder=exp_folder)

            if export['edgelist_table']:
                exportEdgelistTable(supply_chain_network=G, export_folder=exp_folder)

            if export['inventories']:
                exportInventories(firm_list, export_folder=exp_folder)

        obs.disruption_time = disruption['start_time']
        logging.info('Simulation will last ' + str(Tfinal) + ' time steps.')
        logging.info('A disruption will occur at time ' + str(disruption['start_time']) + ', it will affect ' +
                     str(len(disruption['node'])) + ' nodes and ' +
                     str(len(disruption['edge'])) + ' edges for ' +
                     str(disruption['duration']) + ' time steps.')

        logging.info("Starting time loop")
        for t in range(1, Tfinal + 1):
            logging.info('Time t=' + str(t))
            runOneTimeStep(transport_network=T, sc_network=G, firm_list=firm_list,
                           country_list=country_list, household_list=household_list,
                           disruptions=[disruption],
                           congestion=congestion,
                           route_optimization_weight=route_optimization_weight,
                           logistics_modes=logistics_modes,
                           explicit_service_firm=explicit_service_firm,
                           sectors_no_transport_network=sectors_no_transport_network,
                           propagate_input_price_change=propagate_input_price_change,
                           rationing_mode=rationing_mode,
                           observer=obs,
                           time_step=t,
                           export_folder=exp_folder,
                           export_flows=export_flows,
                           flow_types_to_export=flow_types_to_export,
                           transport_edges=transport_edges,
                           export_sc_flow_analysis=False,
                           monetary_units_in_model=monetary_units_in_model,
                           cost_repercussion_mode=cost_repercussion_mode)
            logging.debug('End of t=' + str(t))

            if (t > 1) and epsilon_stop_condition:
                household_extra_spending = sum([household.extra_spending for household in household_list])
                household_consumption_loss = sum([household.consumption_loss for household in household_list])
                country_extra_spending = sum([country.extra_spending for country in country_list])
                country_consumption_loss = sum([country.consumption_loss for country in country_list])
                if (household_extra_spending <= epsilon_stop_condition) & \
                        (household_consumption_loss <= epsilon_stop_condition) & \
                        (country_extra_spending <= epsilon_stop_condition) & \
                        (country_consumption_loss <= epsilon_stop_condition):
                    logging.info(
                        'Household and country extra spending and consumption loss are at pre-disruption value. ' \
                        + "Simulation stops.")
                    break

        computation_time = time.time() - t0
        logging.info("Time loop completed, {:.02f} min".format(computation_time / 60))

        # obs.evaluate_results(T, household_list, disruption, disruption_analysis['duration'],
        #  epsilon_stop_condition, per_firm=export['impact_per_firm'])
        obs.evaluate_results(T, household_list, disruption['node'],
                             epsilon_stop_condition, per_firm=export['impact_per_firm'])

        if export['time_series']:
            exportTimeSeries(obs, exp_folder)

        if export['criticality']:
            writeCriticalityResults(criticality_export_file, obs, disruption,
                                    disruption_analysis['duration'], computation_time)

        if export['impact_per_firm']:
            writeResPerFirmResults(extra_spending_export_file,
                                   missing_consumption_export_file, obs, disruption)

        # if export['agent_data']:
        #     exportAgentData(obs, export_folder=exp_folder)

        del obs

    if export['criticality']:
        criticality_export_file.close()

    logging.info("End of simulation")
