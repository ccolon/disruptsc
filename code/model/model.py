import pandas as pd
import logging
from .caching_functions import \
    load_cached_transport_network, \
    load_cached_agent_data, \
    load_cached_transaction_table, \
    cache_transport_network, \
    cache_agent_data
from .model_parameters import ModelParameters
from .transport_network_builder_functions import \
    create_transport_network
from .agent_builder_functions import \
    filter_sector, \
    define_firms_from_local_economic_data, \
    create_firms, \
    define_firms_from_network_data, \
    extract_final_list_of_sector, \
    define_households, \
    add_households_for_firms, \
    create_households, \
    load_technical_coefficients


class Model(object):
    def __init__(self, country_list: list, household_list:list, firm_list: list):
        # Parameters and filepath
        self.filepaths = None
        self.parameters = None
        # Initialization states
        self.transport_network_initialized = False
        self.agents_initialized = False
        self.sc_network_initialized = False
        self.logistic_routes_initialized = False
        # Agent variables
        self.country_list = None
        self.household_list = None
        self.household_table = None
        self.firm_list = None
        # Transport network variables
        self.transport_edges = None
        self.transport_nodes = None
        self.transport_network = None

    def setup(self, parameters: ModelParameters, filepaths: dict):
        # Parameters
        self.parameters = parameters
        # Filepaths
        self.filepaths = filepaths
        # TODO put here the setup functions

    @classmethod
    def from_parameters(cls, filepaths: dict,
                        parameters: dict,
                        transport_params: dict,
                        transport_modes: list,
                        cached: bool):
        if cached:
            transport_network, transport_nodes, transport_edges = \
                load_cached_transport_network()
        else:
            transport_network, transport_nodes, transport_edges = \
                create_transport_network(transport_modes, filepaths, transport_params)

            data_to_cache = {
                "transport_network": transport_network,
                'transport_edges': transport_edges,
                'transport_nodes': transport_nodes
            }
            cache_transport_network(data_to_cache)

        return cls(...)

    def setup_transport_network(
            self,
            cached: bool,
            filepaths: dict,
            transport_modes: list,
            transport_params: dict,
            route_optimization_weight: str,
            logistics_modes: dict
    ):
        if cached:
            self.transport_network, self.transport_nodes, self.transport_edges = \
                load_cached_transport_network()
        else:
            self.transport_network, self.transport_nodes, self.transport_edges = \
                create_transport_network(transport_modes, filepaths, transport_params)

            data_to_cache = {
                "transport_network": self.transport_network,
                'transport_edges': self.transport_edges,
                'transport_nodes': self.transport_nodes
            }
            cache_transport_network(data_to_cache)

        self.transport_network.defineWeights(route_optimization_weight, logistics_modes)
        self.transport_network.log_km_per_transport_modes()  # Print data on km per mode

        self.transport_network_initialized = True

    def setup_firms(self):
        # TODO write
        pass

    def setup_households(self):
        pass

    def setup_countries(self):
        pass

    def setup_agents(
            self,
            cached: bool,
            filepaths: dict,
            cutoff_sector_output: dict,
            cutoff_sector_demand: dict,
            combine_sector_cutoff: str,
            sectors_to_include: list,
            sectors_to_exclude: list,
            firm_data_type: str,
            reactivity_rate: float,
            utilization_rate: float,
            pop_cutoff: float,
            pop_density_cutoff: float,
            local_demand_cutoff: float,
            transport_nodes: list,
            time_resolution: str,
            monetary_units_in_model: str,
            monetary_units_inputed: str,
            io_cutoff: float
    ):
        if cached:
            self.firm_list, self.household_table, self.household_list, self.country_list = load_cached_agent_data()
            if firm_data_type == "supplier-buyer network":
                transaction_table = load_cached_transaction_table()
            self.transport_network, self.transport_nodes, self.transport_edges = \
                load_cached_transport_network()
        else:

            logging.info('Filtering the sectors based on their output. ' +
                         "Cutoff type is " + cutoff_sector_output['type'] +
                         ", cutoff value is " + str(cutoff_sector_output['value']))
            sector_table = pd.read_csv(filepaths['sector_table'])
            filtered_sectors = filter_sector(sector_table,
                                             cutoff_sector_output=cutoff_sector_output,
                                             cutoff_sector_demand=cutoff_sector_demand,
                                             combine_sector_cutoff=combine_sector_cutoff,
                                             sectors_to_include=sectors_to_include,
                                             sectors_to_exclude=sectors_to_exclude)
            output_selected = sector_table.loc[sector_table['sector'].isin(filtered_sectors), 'output'].sum()
            final_demand_selected = sector_table.loc[
                sector_table['sector'].isin(filtered_sectors), 'final_demand'].sum()
            logging.info(
                str(len(filtered_sectors)) + ' sectors selected over ' + str(sector_table.shape[0]) + ' representing ' +
                "{:.0f}%".format(output_selected / sector_table['output'].sum() * 100) + ' of total output and ' +
                "{:.0f}%".format(final_demand_selected / sector_table['final_demand'].sum() * 100) + ' of final demand'
            )
            logging.info('The filtered sectors are: ' + str(filtered_sectors))

            logging.info('Generating the firms')
            if firm_data_type == "disaggregating IO":
                firm_table, firm_table_per_admin_unit = define_firms_from_local_economic_data(
                    filepath_admin_unit_economic_data=filepaths['admin_unit_data'],
                    sectors_to_include=filtered_sectors,
                    transport_nodes=self.transport_nodes,
                    filepath_sector_table=filepaths['sector_table']
                )
            elif firm_data_type == "supplier-buyer network":
                firm_table = define_firms_from_network_data(
                    filepath_firm_table=filepaths['firm_table'],
                    filepath_location_table=filepaths['location_table'],
                    sectors_to_include=sectors_to_include,
                    transport_nodes=self.transport_nodes,
                    filepath_sector_table=filepaths['sector_table'])
            else:
                raise ValueError(firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'")
            nb_firms = 'all'
            logging.info('Creating firm_list. nb_firms: ' + str(nb_firms) +
                         ' reactivity_rate: ' + str(reactivity_rate) +
                         ' utilization_rate: ' + str(utilization_rate))
            firm_list = create_firms(
                firm_table=firm_table,
                keep_top_n_firms=nb_firms,
                reactivity_rate=reactivity_rate,
                utilization_rate=utilization_rate
            )

            n, present_sectors, flow_types_to_export = extract_final_list_of_sector(firm_list)

            # Create households
            logging.info('Defining the number of households to generate and their purchase plan')
            household_table, household_sector_consumption = define_households(
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
            cond_no_household = ~firm_table['od_point'].isin(household_table['od_point'])
            if cond_no_household.sum() > 0:
                logging.info('We add local households for firms')
                household_table, household_sector_consumption = add_households_for_firms(
                    firm_table=firm_table,
                    household_table=household_table,
                    filepath_adminunit_data=filepaths['adminunit_data'],
                    sector_table=sector_table,
                    filtered_sectors=present_sectors,
                    time_resolution=time_resolution,
                    target_units=monetary_units_in_model,
                    input_units=monetary_units_inputed
                )
            household_list = create_households(
                household_table=household_table,
                household_sector_consumption=household_sector_consumption
            )

            # Loading the technical coefficients
            if firm_data_type == "disaggregating IO":
                import_code = sector_table.loc[sector_table['type'] == 'imports', 'sector'].iloc[0]  # usually it is IMP
                firm_list = load_technical_coefficients(firm_list, filepaths['tech_coef'], io_cutoff, import_code)

            elif firm_data_type == "supplier-buyer network":
                firm_table, transaction_table = calibrate_input_mix(
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

        # Locate firms and households on transport network
        self.transport_network.locate_firms_on_nodes(firm_list, transport_nodes)
        self.transport_network.locate_households_on_nodes(household_list, transport_nodes)
