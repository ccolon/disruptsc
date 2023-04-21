import networkx as nx
import numpy as np
import pandas as pd
import logging
from .caching_functions import \
    load_cached_transport_network, \
    load_cached_agent_data, \
    load_cached_transaction_table, \
    cache_transport_network, \
    cache_agent_data, load_cached_sc_network, cache_sc_network, load_cached_logistic_routes, cache_logistic_routes
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
    load_technical_coefficients, calibrate_input_mix, load_inventories, create_countries, load_ton_usd_equivalence
from code.parameters import Parameters


class Model(object):
    def __init__(self, parameters: Parameters):
        # Parameters and filepath
        self.parameters = parameters
        # Initialization states
        self.transport_network_initialized = False
        self.agents_initialized = False
        self.sc_network_initialized = False
        self.logistic_routes_initialized = False
        # Main reference table
        self.sector_table = None
        # Transport network variables
        self.transport_edges = None
        self.transport_nodes = None
        self.transport_network = None
        # Agent variables
        self.firm_list = None
        self.firm_table = None
        self.household_list = None
        self.household_table = None
        self.country_list = None
        self.transaction_table = None
        # Supply-chain network variables
        self.sc_network = None

    def is_initialized(self):
        if all([self.transport_network_initialized, self.agents_initialized,
                self.sc_network_initialized, self.logistic_routes_initialized]):
            return True
        else:
            return False

    def setup_transport_network(self, cached: bool):
        if cached:
            self.transport_network, self.transport_nodes, self.transport_edges = \
                load_cached_transport_network()
        else:
            self.transport_network, self.transport_nodes, self.transport_edges = \
                create_transport_network(
                    transport_modes=self.parameters.transport_modes,
                    filepaths=self.parameters.filepaths
                )

            data_to_cache = {
                "transport_network": self.transport_network,
                'transport_nodes': self.transport_nodes,
                'transport_edges': self.transport_edges
            }
            cache_transport_network(data_to_cache)

        self.transport_network.defineWeights(
            route_optimization_weight=self.parameters.route_optimization_weight,
            logistics_modes=self.parameters.logistics_modes
        )
        self.transport_network.log_km_per_transport_modes()  # Print data on km per mode
        self.transport_network_initialized = True

    def setup_firms(self):
        # TODO write
        pass

    def setup_households(self):
        pass

    def setup_countries(self):
        pass

    def setup_agents(self, cached: bool):
        if cached:
            self.sector_table, self.firm_list, self.firm_table, self.household_list, self.household_table, \
                self.country_list = load_cached_agent_data()
            if self.parameters.firm_data_type == "supplier-buyer network":
                self.transaction_table = load_cached_transaction_table()
        else:
            logging.info('Filtering the sectors based on their output. ' +
                         "Cutoff type is " + self.parameters.cutoff_sector_output['type'] +
                         ", cutoff value is " + str(self.parameters.cutoff_sector_output['value']))
            self.sector_table = pd.read_csv(self.parameters.filepaths['sector_table'])
            filtered_sectors = filter_sector(self.sector_table,
                                             cutoff_sector_output=self.parameters.cutoff_sector_output,
                                             cutoff_sector_demand=self.parameters.cutoff_sector_demand,
                                             combine_sector_cutoff=self.parameters.combine_sector_cutoff,
                                             sectors_to_include=self.parameters.sectors_to_include,
                                             sectors_to_exclude=self.parameters.sectors_to_exclude)
            output_selected = self.sector_table.loc[self.sector_table['sector'].isin(filtered_sectors), 'output'].sum()
            final_demand_selected = self.sector_table.loc[
                self.sector_table['sector'].isin(filtered_sectors), 'final_demand'].sum()
            logging.info(
                str(len(filtered_sectors)) + ' sectors selected over ' + str(
                    self.sector_table.shape[0]) + ' representing ' +
                "{:.0f}%".format(output_selected / self.sector_table['output'].sum() * 100) + ' of total output and ' +
                "{:.0f}%".format(
                    final_demand_selected / self.sector_table['final_demand'].sum() * 100) + ' of final demand'
            )
            logging.info('The filtered sectors are: ' + str(filtered_sectors))

            logging.info('Generating the firms')
            if self.parameters.firm_data_type == "disaggregating IO":
                self.firm_table, firm_table_per_admin_unit = define_firms_from_local_economic_data(
                    filepath_admin_unit_economic_data=self.parameters.filepaths['admin_unit_data'],
                    sectors_to_include=filtered_sectors,
                    transport_nodes=self.transport_nodes,
                    filepath_sector_table=self.parameters.filepaths['sector_table']
                )
            elif self.parameters.firm_data_type == "supplier-buyer network":
                self.firm_table = define_firms_from_network_data(
                    filepath_firm_table=self.parameters.filepaths['firm_table'],
                    filepath_location_table=self.parameters.filepaths['location_table'],
                    sectors_to_include=filtered_sectors,
                    transport_nodes=self.transport_nodes,
                    filepath_sector_table=self.parameters.filepaths['sector_table'])
            else:
                raise ValueError(
                    self.parameters.firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'"
                )
            nb_firms = 'all'  # Weird
            logging.info('Creating firm_list. nb_firms: ' + str(nb_firms) +
                         ' reactivity_rate: ' + str(self.parameters.reactivity_rate) +
                         ' utilization_rate: ' + str(self.parameters.utilization_rate))
            self.firm_list = create_firms(
                firm_table=self.firm_table,
                keep_top_n_firms=nb_firms,
                reactivity_rate=self.parameters.reactivity_rate,
                utilization_rate=self.parameters.utilization_rate
            )

            n, present_sectors, flow_types_to_export = extract_final_list_of_sector(self.firm_list)

            # Create households
            logging.info('Defining the number of households to generate and their purchase plan')
            self.household_table, household_sector_consumption = define_households(
                sector_table=self.sector_table,
                filepath_admin_unit_data=self.parameters.filepaths['admin_unit_data'],
                filtered_sectors=present_sectors,
                pop_cutoff=self.parameters.pop_cutoff,
                pop_density_cutoff=self.parameters.pop_density_cutoff,
                local_demand_cutoff=self.parameters.local_demand_cutoff,
                transport_nodes=self.transport_nodes,
                time_resolution=self.parameters.time_resolution,
                target_units=self.parameters.monetary_units_in_model,
                input_units=self.parameters.monetary_units_inputed
            )
            cond_no_household = ~self.firm_table['od_point'].isin(self.household_table['od_point'])
            if cond_no_household.sum() > 0:
                logging.info('We add local households for firms')
                self.household_table, household_sector_consumption = add_households_for_firms(
                    firm_table=self.firm_table,
                    household_table=self.household_table,
                    filepath_admin_unit_data=self.parameters.filepaths['admin_unit_data'],
                    sector_table=self.sector_table,
                    filtered_sectors=present_sectors,
                    time_resolution=self.parameters.time_resolution,
                    target_units=self.parameters.monetary_units_in_model,
                    input_units=self.parameters.monetary_units_inputed
                )
            self.household_list = create_households(
                household_table=self.household_table,
                household_sector_consumption=household_sector_consumption
            )

            # Loading the technical coefficients
            if self.parameters.firm_data_type == "disaggregating IO":
                import_code_in_table = self.sector_table.loc[self.sector_table['type'] == 'imports', 'sector'].iloc[
                    0]  # usually it is IMP
                self.firm_list = load_technical_coefficients(
                    self.firm_list, self.parameters.filepaths['tech_coef'], self.parameters.io_cutoff,
                    import_code_in_table
                )

            elif self.parameters.firm_data_type == "supplier-buyer network":
                self.firm_table, self.transaction_table = calibrate_input_mix(
                    firm_list=self.firm_list,
                    firm_table=self.firm_table,
                    sector_table=self.sector_table,
                    filepath_transaction_table=self.parameters.filepaths['transaction_table']
                )

            else:
                raise ValueError(
                    self.parameters.firm_data_type + " should be one of 'disaggregating', 'supplier-buyer network'"
                )

            # Loading the inventories
            self.firm_list = load_inventories(
                firm_list=self.firm_list,
                inventory_duration_target=self.parameters.inventory_duration_target,
                filepath_inventory_duration_targets=self.parameters.filepaths['inventory_duration_targets'],
                extra_inventory_target=self.parameters.extra_inventory_target,
                inputs_with_extra_inventories=self.parameters.inputs_with_extra_inventories,
                buying_sectors_with_extra_inventories=self.parameters.buying_sectors_with_extra_inventories,
                min_inventory=1
            )

            # Create agents: Countries
            self.country_list = create_countries(
                filepath_imports=self.parameters.filepaths['imports'],
                filepath_exports=self.parameters.filepaths['exports'],
                filepath_transit_matrix=self.parameters.filepaths['transit_matrix'],
                transport_nodes=self.transport_nodes,
                present_sectors=present_sectors,
                countries_to_include=self.parameters.countries_to_include,
                time_resolution=self.parameters.time_resolution,
                target_units=self.parameters.monetary_units_in_model,
                input_units=self.parameters.monetary_units_inputed
            )

            # Specify the weight of a unit worth of good, which may differ according to sector, or even to each
            # firm/countries Note that for imports, i.e. for the goods delivered by a country, and for transit flows,
            # we do not disentangle sectors In this case, we use an average.
            self.firm_list, self.country_list, sector_to_usdPerTon = load_ton_usd_equivalence(
                sector_table=self.sector_table,
                firm_list=self.firm_list,
                country_list=self.country_list
            )

            # Save to tmp folder
            data_to_cache = {
                "sector_table": self.sector_table,
                'firm_table': self.firm_table,
                'present_sectors': present_sectors,
                'flow_types_to_export': flow_types_to_export,
                'firm_list': self.firm_list,
                'household_table': self.household_table,
                'household_list': self.household_list,
                'country_list': self.country_list
            }
            if self.parameters.firm_data_type == "supplier-buyer network":
                data_to_cache['transaction_table'] = self.transaction_table
            cache_agent_data(data_to_cache)

        # Locate firms and households on transport network
        self.transport_network.locate_firms_on_nodes(self.firm_list, self.transport_nodes)
        self.transport_network.locate_households_on_nodes(self.household_list, self.transport_nodes)
        self.agents_initialized = True

    def setup_sc_network(self, cached: bool):
        if cached:
            self.sc_network, self.firm_list, self.household_list, self.country_list = load_cached_sc_network()

        else:
            logging.info(
                f'The supply chain graph is being created. nb_suppliers_per_input: '
                f'{self.parameters.nb_suppliers_per_input}')
            self.sc_network = nx.DiGraph()

            logging.info('Households are selecting their retailers (domestic B2C flows)')
            for household in self.household_list:
                household.select_suppliers(self.sc_network, self.firm_list, self.firm_table,
                                           self.parameters.nb_suppliers_per_input,
                                           self.parameters.weight_localization_household)

            logging.info('Exporters are being selected by purchasing countries (export B2B flows)')
            logging.info('and trading countries are being connected (transit flows)')
            for country in self.country_list:
                country.select_suppliers(self.sc_network, self.firm_list, self.country_list,
                                         self.sector_table, self.transport_nodes)

            logging.info(
                f'Firms are selecting their domestic and international suppliers (import B2B flows) '
                f'(domestic B2B flows). Weight localisation is {self.parameters.weight_localization_firm}'
            )
            import_code_from_table = self.sector_table.loc[self.sector_table['type'] == 'imports', 'sector'].iloc[0]

            if self.parameters.firm_data_type == "disaggregating IO":
                for firm in self.firm_list:
                    firm.select_suppliers(self.sc_network, self.firm_list, self.country_list,
                                          self.parameters.nb_suppliers_per_input,
                                          self.parameters.weight_localization_firm,
                                          import_code=import_code_from_table)

            elif self.parameters.firm_data_type == "supplier-buyer network":
                for firm in self.firm_list:
                    inputed_supplier_links = self.transaction_table[self.transaction_table['buyer_id'] == firm.pid]
                    output = self.firm_table.set_index('id').loc[firm.pid, "output"]
                    firm.select_suppliers_from_data(self.sc_network, self.firm_list, self.country_list,
                                                    inputed_supplier_links, output,
                                                    import_code=import_code_from_table)

            else:
                raise ValueError(self.parameters.firm_data_type +
                                 " should be one of 'disaggregating', 'supplier-buyer network'")

            logging.info('The nodes and edges of the supplier--buyer have been created')
            # Save to tmp folder
            data_to_cache = {
                "supply_chain_network": self.sc_network,
                'firm_list': self.firm_list,
                'household_list': self.household_list,
                'country_list': self.country_list
            }
            cache_sc_network(data_to_cache)

            self.sc_network_initialized = True

    def setup_logistic_routes(self, cached: bool):
        if cached:
            self.sc_network, self.transport_network, self.firm_list, self.household_list, \
                self.country_list = load_cached_logistic_routes()

        else:
            logging.info('The supplier--buyer graph is being connected to the transport network')
            logging.info('Each B2B and transit edge is being linked to a route of the transport network')
            transport_modes = pd.read_csv(self.parameters.filepaths['transport_modes'])
            logging.info('Routes for transit and import flows are being selected by trading countries')
            for country in self.country_list:
                country.choose_initial_routes(self.sc_network, self.transport_network, transport_modes,
                                              self.parameters.account_capacity,
                                              self.parameters.monetary_units_in_model)
            logging.info('Routes for exports and B2B domestic flows are being selected by domestic firms')
            for firm in self.firm_list:
                if firm.sector_type not in self.parameters.sectors_no_transport_network:
                    firm.choose_initial_routes(self.sc_network, self.transport_network, transport_modes,
                                               self.parameters.account_capacity,
                                               self.parameters.monetary_units_in_model)
            # Save to tmp folder
            data_to_cache = {
                'transport_network': self.transport_network,
                "supply_chain_network": self.sc_network,
                'firm_list': self.firm_list,
                'household_list': self.household_list,
                'country_list': self.country_list
            }
            cache_logistic_routes(data_to_cache)

            self.logistic_routes_initialized = True

    def reset_variables(self):
        logging.info("Resetting variables on transport network")
        self.transport_network.reinitialize_flows_and_disruptions()

        logging.info("Resetting agents and commercial links variables")
        for household in self.firm_list:
            household.reset_variables()
            for edge in self.sc_network.in_edges(household):
                self.sc_network[edge[0]][household]['object'].reset_variables()
        for firm in self.firm_list:
            firm.reset_variables()
            for edge in self.sc_network.in_edges(firm):
                self.sc_network[edge[0]][firm]['object'].reset_variables()
        for country in self.country_list:
            country.reset_variables()
            for edge in self.sc_network.in_edges(country):
                self.sc_network[edge[0]][country]['object'].reset_variables()

    def set_initial_conditions(self):
        logging.info("Setting initial conditions to input-output equilibrium")
        """
        Initialize the supply chain network at the input--output equilibrium
    
        We will use the matrix forms to solve the following equation for X (production):
        D + E + AX = X + I
        where:
            D: final demand from households
            E: exports
            I: imports
            X: firm productions
            A: the input-output matrix
        These vectors and matrices are in the firm-and-country space.
        """

        # Get the weighted connectivity matrix.
        # Weight is the sectoral technical coefficient, if there is only one supplier for the input
        # It there are several, the technical coefficient is multiplied by the share of input of
        # this type that the firm buys to this supplier.
        firm_connectivity_matrix = nx.adjacency_matrix(
            self.sc_network,
            # graph.subgraph(list(graph.nodes)[:-1]),
            weight='weight',
            nodelist=self.firm_list
        ).todense()
        # Imports are considered as "a sector". We get the weight per firm for these inputs.
        # TODO !!! aren't I computing the same thing as the IMP tech coef? To check
        import_weight_per_firm = [
            sum([
                self.sc_network[supply_edge[0]][supply_edge[1]]['weight']
                for supply_edge in self.sc_network.in_edges(firm)
                if self.sc_network[supply_edge[0]][supply_edge[1]]['object'].category == 'import'
            ])
            for firm in self.firm_list
        ]
        n = len(self.firm_list)

        # Build final demand vector per firm, of length n
        # Exports are considered as final demand
        final_demand_vector = self.build_final_demand_vector(self.household_list, self.country_list, self.firm_list)

        # Solve the input--output equation
        eq_production_vector = np.linalg.solve(
            np.eye(n) - firm_connectivity_matrix,
            final_demand_vector
        )

        # Initialize households variables
        for household in self.household_list:
            household.initialize_var_on_purchase_plan()

        # Compute costs
        # 1. Input costs
        domestic_input_cost_vector = np.multiply(
            firm_connectivity_matrix.sum(axis=0).reshape((n, 1)),
            eq_production_vector
        )
        import_input_cost_vector = np.multiply(
            np.array(import_weight_per_firm).reshape((n, 1)),
            eq_production_vector
        )
        input_cost_vector = domestic_input_cost_vector + import_input_cost_vector
        # 2. Transport costs
        proportion_of_transport_cost_vector = 0.2 * np.ones((n, 1))  # TODO should be parametrized
        transport_cost_vector = np.multiply(eq_production_vector, proportion_of_transport_cost_vector)
        # 3. Compute other costs based on margin
        margin = np.array([firm.target_margin for firm in self.firm_list]).reshape((n, 1))
        other_cost_vector = np.multiply(eq_production_vector, (1 - margin)) - input_cost_vector - transport_cost_vector

        # Based on these calculus, update agents variables
        # 1. Firm operational variables
        for firm in self.firm_list:  # TODO make it a FirmCollection method
            firm.initialize_ope_var_using_eq_production(
                eq_production=eq_production_vector[(firm.pid, 0)]
            )
        # 2. Firm financial variables
        for firm in self.firm_list:
            firm.initialize_fin_var_using_eq_cost(
                eq_production=eq_production_vector[(firm.pid, 0)],
                eq_input_cost=input_cost_vector[(firm.pid, 0)],
                eq_transport_cost=transport_cost_vector[(firm.pid, 0)],
                eq_other_cost=other_cost_vector[(firm.pid, 0)]
            )
        # 3. Commercial links: agents set their order
        for household in self.household_list:
            household.send_purchase_orders(self.sc_network)
        for country in self.country_list:
            country.send_purchase_orders(self.sc_network)
        for firm in self.firm_list:
            firm.send_purchase_orders(self.sc_network)
        # 4. The following is just to set once for all the share of sales of each client
        for firm in self.firm_list:
            firm.retrieve_orders(self.sc_network)
            firm.aggregate_orders(print_info=True)
            firm.eq_total_order = firm.total_order
            firm.calculate_client_share_in_sales()

        # Set price to 1
        self.reset_prices()

    def reset_prices(self):
        # set prices to 1
        for edge in self.sc_network.edges:
            self.sc_network[edge[0]][edge[1]]['object'].price = 1

    @staticmethod
    def build_final_demand_vector(household_list: list, country_list: list, firm_list: list) -> np.array:
        """
        Create a numpy.Array of the final demand per firm, including exports

        Households and countries should already have set their purchase plan

        Returns
        -------
        numpy.Array of dimension (len(firm_list), 1)
        """
        final_demand_vector = np.zeros((len(firm_list), 1))

        # Collect households final demand. They buy only from firms.
        for household in household_list:
            for retailer_id, quantity in household.purchase_plan.items():
                final_demand_vector[(retailer_id, 0)] += quantity

        # Collect country final demand. They buy from firms and countries.
        # We need to filter the demand directed to firms only.
        for country in country_list:
            for supplier_id, quantity in country.purchase_plan.items():
                if isinstance(supplier_id, int):  # we only consider purchase from firms, not from other countries
                    final_demand_vector[(supplier_id, 0)] += quantity

        return final_demand_vector
