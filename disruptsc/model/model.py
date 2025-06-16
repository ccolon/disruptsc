from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import logging

import pandas as pd
from tqdm import tqdm

from .caching_functions import \
    load_cached_transport_network, \
    load_cached_agent_data, \
    load_cached_transaction_table, \
    cache_transport_network, \
    cache_agent_data, load_cached_sc_network, cache_sc_network, load_cached_logistic_routes, cache_logistic_routes, \
    cache_model
from disruptsc.model.check_functions import compare_production_purchase_plans
from disruptsc.model.country_builder_functions import create_countries_from_mrio, create_countries
from disruptsc.model.firm_builder_functions import \
    define_firms_from_network_data, \
    define_firms_from_mrio, create_firms, calibrate_input_mix, load_mrio_tech_coefs, \
    load_inventories
from disruptsc.model.household_builder_functions import define_households_from_mrio, define_households, \
    add_households_for_firms, \
    create_households
from disruptsc.model.transport_network_builder_functions import \
    create_transport_network
from disruptsc.model.builder_functions import filter_sector
from disruptsc.parameters import Parameters
from disruptsc.disruption.disruption import DisruptionList, TransportDisruption, CapitalDestruction, Recovery
from disruptsc.simulation.simulation import Simulation
from disruptsc.network.sc_network import ScNetwork
from disruptsc.network.mrio import Mrio

if TYPE_CHECKING:
    from disruptsc.agents.country import Countries
    from disruptsc.agents.firm import Firms
    from disruptsc.agents.household import Households


class Model(object):
    def __init__(self, parameters: Parameters):
        # Parameters and filepath
        self.mrio = None
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
        self.firms = None
        self.firm_table = None
        self.households = None
        self.household_table = None
        self.countries = None
        self.country_table = None
        self.transaction_table = None
        self.commercial_link_table = None
        # Supply-chain network variables
        self.sc_network = None
        # Disruption variable
        self.disruption_list = None
        self.reconstruction_market = None

    @property
    def is_initialized(self):
        if all([self.transport_network_initialized, self.agents_initialized,
                self.sc_network_initialized, self.logistic_routes_initialized]):
            return True
        else:
            return False

    def setup_transport_network(self, cached: bool = False, with_transport: bool = True):
        if cached:
            self.transport_network, self.transport_edges, self.transport_nodes = \
                load_cached_transport_network()
        else:
            self.transport_network, self.transport_edges, self.transport_nodes = \
                create_transport_network(
                    transport_modes=self.parameters.transport_modes,
                    filepaths=self.parameters.filepaths,
                    logistics_parameters=self.parameters.logistics,
                    time_resolution=self.parameters.time_resolution,
                    admin=self.parameters.admin
                )

            data_to_cache = {
                "transport_network": self.transport_network,
                'transport_edges': self.transport_edges,
                'transport_nodes': self.transport_nodes
            }
            cache_transport_network(data_to_cache)

        # self.transport_network.define_weights(
        #     route_optimization_weight=self.parameters.route_optimization_weight
        # )
        self.parameters.add_variability_to_basic_cost()
        self.transport_network.ingest_logistic_data(self.parameters.logistics)
        self.transport_network.log_km_per_transport_modes()  # Print data on km per mode
        self.transport_network_initialized = True

    def shuffle_logistic_costs(self):
        self.parameters.add_variability_to_basic_cost()
        self.transport_network.ingest_logistic_data(self.parameters.logistics)

    def setup_firms(self):
        pass

    def setup_households(self):
        pass

    def setup_countries(self):
        pass

    def setup_agents(self, cached: bool = False):
        if cached:
            self.mrio, self.sector_table, self.firms, self.firm_table, self.households, self.household_table, \
                self.countries = load_cached_agent_data()
            if self.parameters.firm_data_type == "supplier-buyer network":
                self.transaction_table = load_cached_transaction_table()
            self.mrio = Mrio(self.mrio, monetary_units=self.parameters.monetary_units_in_data)
        else:
            logging.info(f"Filtering the sectors based on their output. "
                         f"Cutoff type is {self.parameters.cutoff_sector_output['type']}, "
                         f"cutoff value is {self.parameters.cutoff_sector_output['value']}")
            self.mrio = Mrio.load_mrio_from_filepath(self.parameters.filepaths['mrio'],
                                                     self.parameters.monetary_units_in_data)
            self.sector_table = pd.read_csv(self.parameters.filepaths['sector_table'])

            filtered_industries = filter_sector(self.mrio,
                                                cutoff_sector_output=self.parameters.cutoff_sector_output,
                                                cutoff_sector_demand=self.parameters.cutoff_sector_demand,
                                                combine_sector_cutoff=self.parameters.combine_sector_cutoff,
                                                sectors_to_include=self.parameters.sectors_to_include,
                                                sectors_to_exclude=self.parameters.sectors_to_exclude,
                                                monetary_units_in_data=self.parameters.monetary_units_in_data)
            output_selected = self.mrio.get_total_output_per_region_sectors(filtered_industries).sum()
            output_total = self.mrio.get_total_output_per_region_sectors().sum()
            final_demand_selected = self.mrio.get_final_demand(filtered_industries).sum(axis=1).sum()
            final_demand_total = self.mrio.get_final_demand().sum(axis=1).sum()
            # final_demand_selected = self.sector_table.loc[
            #     self.sector_table['sector'].isin(filtered_industries), 'final_demand'].sum()
            logging.info(f"{len(filtered_industries)} sectors selected over {len(self.mrio.region_sectors)} "
                         f"covering {output_selected / output_total:.0%} of total output "
                         f"& {final_demand_selected / final_demand_total:.0%} of final demand")
            logging.info(f'The filtered sectors are: {filtered_industries}')

            logging.info('Generating the firms')
            if self.parameters.firm_data_type == "mrio":
                self.firm_table = define_firms_from_mrio(self.mrio,
                                                         filepath_sectors=self.parameters.filepaths['sector_table'],
                                                         filepath_regions=self.parameters.filepaths['region_table'],
                                                         path_disag=self.parameters.filepaths['disag'],
                                                         transport_nodes=self.transport_nodes,
                                                         io_cutoff=self.parameters.io_cutoff,
                                                         cutoff_firm_output=self.parameters.cutoff_firm_output,
                                                         monetary_units_in_data=self.parameters.monetary_units_in_data,
                                                         admin=self.parameters.admin)
            elif self.parameters.firm_data_type == "supplier-buyer network":
                self.firm_table = define_firms_from_network_data(
                    filepath_firm_table=self.parameters.filepaths['firm_table'],
                    filepath_location_table=self.parameters.filepaths['location_table'],
                    sectors_to_include=filtered_industries,
                    transport_nodes=self.transport_nodes,
                    filepath_sector_table=self.parameters.filepaths['sector_table'])
            else:
                raise ValueError(f"Unknown firm_data_type: {self.parameters.firm_data_type}. "
                               f"Supported types: 'mrio', 'supplier-buyer network'")
            nb_firms = 'all'  # Weird
            logging.info(f"Creating firms. nb_firms: {nb_firms} "
                         f"inventory_restoration_time: {self.parameters.inventory_restoration_time} "
                         f"utilization_rate: {self.parameters.utilization_rate}")
            self.firms = create_firms(
                firm_table=self.firm_table,
                keep_top_n_firms=nb_firms,
                inventory_restoration_time=self.parameters.inventory_restoration_time,
                utilization_rate=self.parameters.utilization_rate,
                capital_to_value_added_ratio=self.parameters.capital_to_value_added_ratio,
                admin=self.parameters.admin
            )

            present_sectors, present_region_sectors, flow_types_to_export = self.firms.extract_sectors()

            # Create households
            logging.info('Defining the number of households to generate and their purchase plan')
            if self.parameters.firm_data_type == "mrio":
                self.household_table, household_sector_consumption = define_households_from_mrio(
                    mrio=self.mrio,
                    filepath_households_spatial=self.parameters.filepaths['households_spatial'],
                    transport_nodes=self.transport_nodes,
                    time_resolution=self.parameters.time_resolution,
                    target_units=self.parameters.monetary_units_in_model,
                    input_units=self.parameters.monetary_units_in_data,
                    final_demand_cutoff=self.parameters.cutoff_household_demand,
                    present_region_sectors=present_region_sectors,
                    admin=self.parameters.admin
                )
                print()
            else:
                self.household_table, household_sector_consumption = define_households(
                    sector_table=self.sector_table,
                    filepath_region_data=self.parameters.filepaths['region_data'],
                    filtered_sectors=present_sectors,
                    pop_cutoff=self.parameters.pop_cutoff,
                    pop_density_cutoff=self.parameters.pop_density_cutoff,
                    local_demand_cutoff=self.parameters.local_demand_cutoff,
                    transport_nodes=self.transport_nodes,
                    time_resolution=self.parameters.time_resolution,
                    target_units=self.parameters.monetary_units_in_model,
                    input_units=self.parameters.monetary_units_in_data
                )
                cond_no_household = ~self.firm_table['od_point'].isin(self.household_table['od_point'])
                if cond_no_household.sum() > 0:
                    logging.info('We add local households for firms')
                    self.household_table, household_sector_consumption = add_households_for_firms(
                        firm_table=self.firm_table,
                        household_table=self.household_table,
                        filepath_region_data=self.parameters.filepaths['region_data'],
                        sector_table=self.sector_table,
                        filtered_sectors=present_sectors,
                        time_resolution=self.parameters.time_resolution,
                        target_units=self.parameters.monetary_units_in_model,
                        input_units=self.parameters.monetary_units_in_data
                    )
            self.households = create_households(
                household_table=self.household_table,
                household_sector_consumption=household_sector_consumption,
                admin=self.parameters.admin
            )

            # Load technical coefficients based on firm data type
            if self.parameters.firm_data_type == "mrio":
                load_mrio_tech_coefs(
                    firms=self.firms,
                    mrio=self.mrio,
                    io_cutoff=self.parameters.io_cutoff,
                    monetary_units_in_data=self.parameters.monetary_units_in_data
                )
            elif self.parameters.firm_data_type == "supplier-buyer network":
                self.firms, self.transaction_table = calibrate_input_mix(
                    firms=self.firms,
                    firm_table=self.firm_table,
                    sector_table=self.sector_table,
                    filepath_transaction_table=self.parameters.filepaths['transaction_table']
                )

            else:
                raise ValueError(
                    f"Unknown firm_data_type: {self.parameters.firm_data_type}. "
                    f"Supported types: 'mrio', 'supplier-buyer network'"
                )

            # Loading the inventories
            load_inventories(
                firms=self.firms,
                inventory_duration_targets=self.parameters.inventory_duration_targets,
                model_time_unit=self.parameters.time_resolution,
                sector_table=self.sector_table
            )

            # Create agents: Countries
            if self.parameters.firm_data_type == "mrio":
                self.countries, self.country_table = create_countries_from_mrio(
                    mrio=self.mrio,
                    transport_nodes=self.transport_nodes,
                    filepath_countries_spatial=self.parameters.filepaths['countries_spatial'],
                    filepath_sectors=self.parameters.filepaths['sector_table'],
                    time_resolution=self.parameters.time_resolution,
                    target_units=self.parameters.monetary_units_in_model,
                    input_units=self.parameters.monetary_units_in_data
                )
            else:
                self.countries = create_countries(
                    filepath_imports=self.parameters.filepaths['imports'],
                    filepath_exports=self.parameters.filepaths['exports'],
                    filepath_transit=self.parameters.filepaths['transit'],
                    transport_nodes=self.transport_nodes,
                    present_sectors=present_sectors,
                    countries_to_include=self.parameters.countries_to_include,
                    time_resolution=self.parameters.time_resolution,
                    target_units=self.parameters.monetary_units_in_model,
                    input_units=self.parameters.monetary_units_in_data
                )

            # Save to tmp folder
            data_to_cache = {
                "mrio": self.mrio,
                "sector_table": self.sector_table,
                'firm_table': self.firm_table,
                'present_sectors': present_sectors,
                'present_region_sectors': present_region_sectors,
                'flow_types_to_export': flow_types_to_export,
                'firms': self.firms,
                'household_table': self.household_table,
                'households': self.households,
                'countries': self.countries
            }
            if self.parameters.firm_data_type == "supplier-buyer network":
                data_to_cache['transaction_table'] = self.transaction_table
            cache_agent_data(data_to_cache)

        # Locate firms and households on transport network
        self.transport_network.locate_firms_on_nodes(self.firms)
        self.transport_network.locate_households_on_nodes(self.households)
        self.agents_initialized = True

    def setup_sc_network(self, cached: bool = False):
        if cached:
            self.sc_network, self.firms, self.households, self.countries = load_cached_sc_network()

        else:
            logging.info(
                f'The supply chain graph is being created. nb_suppliers_per_input: '
                f'{self.parameters.nb_suppliers_per_input}')
            self.sc_network = ScNetwork()

            logging.info('Households are selecting their retailers (domestic B2C flows and import B2C flows)')
            for household in tqdm(self.households.values(), total=len(self.households)):
                household.select_suppliers(self.sc_network, self.firms, self.countries,
                                           self.parameters.weight_localization_household,
                                           self.parameters.nb_suppliers_per_input,
                                           self.parameters.logistics['sector_types_to_shipment_method'],
                                           import_label=self.mrio.import_label)

            logging.info('Exporters are being selected by purchasing countries (export B2B flows)')
            logging.info('and trading countries are being connected (transit flows)')

            for country in tqdm(self.countries.values(), total=len(self.countries)):
                country.select_suppliers(self.sc_network, self.firms, self.countries, self.sector_table,
                                         self.parameters.logistics['sector_types_to_shipment_method'])

            logging.info(
                f'Firms are selecting their domestic and international suppliers (import B2B flows) '
                f'(domestic B2B flows). Weight localisation is {self.parameters.weight_localization_firm}'
            )

            if self.parameters.firm_data_type == "mrio":
                for firm in tqdm(self.firms.values(), total=len(self.firms)):
                    firm.select_suppliers(self.sc_network, self.firms, self.countries,
                                          self.parameters.nb_suppliers_per_input,
                                          self.parameters.weight_localization_firm,
                                          self.parameters.logistics['sector_types_to_shipment_method'],
                                          import_label=self.mrio.import_label)

            elif self.parameters.firm_data_type == "supplier-buyer network":
                for firm in self.firms.values():
                    inputed_supplier_links = self.transaction_table[self.transaction_table['buyer_id'] == firm.pid]
                    output = self.firm_table.set_index('id').loc[firm.pid, "output"]
                    firm.select_suppliers_from_data(self.sc_network, self.firms, self.countries,
                                                    inputed_supplier_links, output,
                                                    import_code='IMP')

            else:
                raise ValueError(f"Unknown firm_data_type: {self.parameters.firm_data_type}. "
                               f"Supported types: 'mrio', 'supplier-buyer network'")

            unconnected_nodes = self.sc_network.identify_disconnected_nodes(self.firms, self.countries, self.households)
            if len(unconnected_nodes) > 0:
                for agent_type, unconnected_node_ids in unconnected_nodes.items():
                    logging.warning(f"{len(unconnected_node_ids)} {agent_type} are not in the sc network: "
                                    f"they have no suppliers, no clients. We remove them.")
                    if agent_type == "firms":
                        for unconnected_firm_id in unconnected_node_ids:
                            # self.sc_network.add_node(self.firms[unconnected_firm_id])
                            del self.firms[unconnected_firm_id]
                    if agent_type == "countries":
                        for unconnected_country_id in unconnected_node_ids:
                            del self.countries[unconnected_country_id]
                    if agent_type == "households":
                        for unconnected_household_id in unconnected_node_ids:
                            del self.households[unconnected_household_id]

            for _ in range(10):
                self.sc_network.remove_useless_commercial_links()

            logging.info(f'Nb of commercial links: {self.sc_network.number_of_edges()}')

            # connected_countries = [node.pid for node in self.sc_network.nodes if node.agent_type == "country"]
            # unconnected_countries = set(self.countries) - set(connected_countries)
            # for unconnected_country in unconnected_countries:
            #     logging.info(f"Country {unconnected_country} is not connected, removing it")
            #     del self.countries[unconnected_country]

            logging.info('The nodes and edges of the supplier--buyer have been created')
            # Save to tmp folder
            data_to_cache = {
                "supply_chain_network": self.sc_network,
                'firms': self.firms,
                'households': self.households,
                'countries': self.countries
            }
            cache_sc_network(data_to_cache)

            self.sc_network_initialized = True

    def create_commercial_link_table(self):
        commercial_links = list(nx.get_edge_attributes(self.sc_network, "object").values())
        self.commercial_link_table = pd.DataFrame({
            link.pid: {'supplier_id': link.supplier_id, 'buyer_id': link.buyer_id,
                       'product': link.product, 'product_type': link.product_type, 'category': link.category,
                       'shipment_method': link.shipment_method, 'use_transport_network': link.use_transport_network,
                       'from': link.route.transport_nodes[0] if link.use_transport_network else None,
                       'to': link.route.transport_nodes[-1] if link.use_transport_network else None,
                       'main_transport_modes': link.route.transport_modes if link.use_transport_network else None}
            for link in commercial_links
        }).transpose()
        self.commercial_link_table.index.name = "pid"

    def setup_logistic_routes(self, cached: bool = False, with_transport: bool = True):
        if not with_transport:
            self.create_commercial_link_table()
            self.logistic_routes_initialized = True
            return

        if cached:
            self.sc_network, self.transport_network, self.commercial_link_table, self.firms, self.households, \
                self.countries = load_cached_logistic_routes()

        else:
            self.countries.assign_cost_profile(self.parameters.logistics['nb_cost_profiles'])
            self.firms.assign_cost_profile(self.parameters.logistics['nb_cost_profiles'])

            logging.info('The supplier--buyer graph is being connected to the transport network')
            logging.info('Each B2B and transit edge_attr is being linked to a route of the transport network')
            logging.info('Routes for transit and import flows are being selected by trading countries')
            self.countries.choose_initial_routes(self.sc_network, self.transport_network,
                                                 self.parameters.capacity_constraint,
                                                 self.parameters.explicit_service_firm,
                                                 self.parameters.transport_to_households,
                                                 self.parameters.sectors_no_transport_network,
                                                 self.parameters.transport_cost_noise_level,
                                                 self.parameters.monetary_units_in_model,
                                                 parallelized=False,
                                                 use_route_cache=self.parameters.use_route_cache)
            logging.info('Routes for exports and B2B domestic flows are being selected by domestic firms')
            self.firms.choose_initial_routes(self.sc_network, self.transport_network,
                                             self.parameters.capacity_constraint,
                                             self.parameters.explicit_service_firm,
                                             self.parameters.transport_to_households,
                                             self.parameters.sectors_no_transport_network,
                                             self.parameters.transport_cost_noise_level,
                                             self.parameters.monetary_units_in_model,
                                             parallelized=False,
                                             use_route_cache=self.parameters.use_route_cache)
            self.create_commercial_link_table()
            # Save to tmp folder
            data_to_cache = {
                'transport_network': self.transport_network,
                "supply_chain_network": self.sc_network,
                'commercial_link_table': self.commercial_link_table,
                'firms': self.firms,
                'households': self.households,
                'countries': self.countries
            }
            cache_logistic_routes(data_to_cache)

            self.logistic_routes_initialized = True

    def reset_variables(self):
        logging.info("Resetting variables on transport network")
        self.transport_network.reinitialize_flows_and_disruptions()

        logging.info("Resetting agents and commercial links variables")
        for commercial_link in nx.get_edge_attributes(self.sc_network, "object").values():
            commercial_link.reset_variables()
        for household in self.households.values():
            household.reset_variables()
        for firm in self.firms.values():
            firm.reset_variables()
        for country in self.countries.values():
            country.reset_variables()

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
        self.reset_variables()
        # Get the weighted connectivity matrix.
        # Weight is the sectoral technical coefficient, if there is only one supplier for the input
        # It there are several, the technical coefficient is multiplied by the share of input of
        # this type that the firm buys to this supplier.
        # l1 = [firm.pid for firm in self.sc_network.nodes if isinstance(firm.pid, int) ]
        # l1.sort()
        # print(l1)
        # print([firm.pid for firm in self.firms])
        firm_connectivity_matrix = nx.adjacency_matrix(
            self.sc_network,
            # graph.subgraph(list(graph.nodes)[:-1]),
            weight='weight',
            nodelist=self.firms.values()
        ).todense()
        # Imports are considered as "a sector". We get the weight per firm for these inputs.
        # TODO !!! aren't I computing the same thing as the IMP tech coef? To check
        import_weight_per_firm = [
            sum([
                self.sc_network[u][v]['weight']
                for u, v in self.sc_network.in_edges(firm)
                if self.sc_network[u][v]['object'].category == 'import'
            ])
            for firm in self.firms.values()
        ]
        transport_input_share_per_firm = self.firms.get_properties('transport_share', "list")
        n = len(self.firms)

        # Build final demand vector per firm, of length n
        # Exports are considered as final demand
        final_demand_vector = self.build_final_demand_vector(self.households, self.countries, self.firms)

        # Solve the input--output equation
        eq_production_vector = np.linalg.solve(
            np.eye(n) - firm_connectivity_matrix,
            final_demand_vector  # + 0.01
        )

        # Initialize households variables
        for household in self.households.values():
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
        # proportion_of_transport_cost_vector = 0.2 * np.ones((n, 1))
        proportion_of_transport_cost_vector = np.array(transport_input_share_per_firm).reshape((n, 1))
        transport_cost_vector = np.multiply(eq_production_vector, proportion_of_transport_cost_vector)
        # 3. Compute other costs based on margin
        margin = np.array([firm.target_margin for firm in self.firms.values()]).reshape((n, 1))
        other_cost_vector = np.multiply(eq_production_vector, (1 - margin)) - input_cost_vector - transport_cost_vector

        # Based on these calculus, update agents variables
        firm_id_to_position_mapping = {firm_id: i for i, firm_id in enumerate(self.firms.get_properties("pid"))}
        # 1. Firm operational variables
        for firm in self.firms.values():  # TODO make it a FirmCollection method
            firm.initialize_operational_variables(
                eq_production=eq_production_vector[(firm_id_to_position_mapping[firm.pid], 0)],
                time_resolution=self.parameters.time_resolution
            )
        # 2. Firm financial variables
        for firm in self.firms.values():
            firm.initialize_financial_variables(
                eq_production=eq_production_vector[(firm_id_to_position_mapping[firm.pid], 0)],
                eq_input_cost=input_cost_vector[(firm_id_to_position_mapping[firm.pid], 0)],
                eq_transport_cost=transport_cost_vector[(firm_id_to_position_mapping[firm.pid], 0)],
                eq_other_cost=other_cost_vector[(firm_id_to_position_mapping[firm.pid], 0)]
            )
        # 3. Commercial links: agents set their order
        for household in self.households.values():
            household.send_purchase_orders(self.sc_network)
        for country in self.countries.values():
            country.send_purchase_orders(self.sc_network)
        for firm in self.firms.values():
            firm.send_purchase_orders(self.sc_network)
        # 4. The following is just to set once for all the share of sales of each client
        for firm in self.firms.values():
            firm.retrieve_orders(self.sc_network)
            firm.aggregate_orders(log_info=True)
            firm.eq_total_order = firm.total_order
            firm.calculate_client_share_in_sales()

        # Set price to 1
        self.reset_prices()

    def reset_prices(self):
        # set prices to 1
        for u, v in self.sc_network.edges:
            self.sc_network[u][v]['object'].price = 1

    @staticmethod
    def build_final_demand_vector(households: "Households", countries: "Countries", firms: "Firms") -> np.array:
        """
        Create a numpy.Array of the final demand per firm, including exports

        Households and countries should already have set their purchase plan

        Returns
        -------
        numpy.Array of dimension (len(firms), 1)
        """
        final_demand_vector = np.zeros((len(firms), 1))

        firm_id_to_position_mapping = {firm_id: i for i, firm_id in enumerate(firms.get_properties("pid"))}

        # Collect households final demand
        for household in households.values():
            for retailer_id, quantity in household.purchase_plan.items():
                if isinstance(retailer_id, int):  # we only consider purchase from firms, not from other countries
                    final_demand_vector[(firm_id_to_position_mapping[retailer_id], 0)] += quantity

        # Collect country final demand
        for country in countries.values():
            for supplier_id, quantity in country.purchase_plan.items():
                if isinstance(supplier_id, int):  # we only consider purchase from firms, not from other countries
                    final_demand_vector[(firm_id_to_position_mapping[supplier_id], 0)] += quantity

        return final_demand_vector

    def run_static(self):
        simulation = Simulation("initial_state")
        logging.info("Simulating the initial state")
        # print("self.production_capacity", self.firms[0].production_capacity)
        self.run_one_time_step(time_step=0, current_simulation=simulation)
        return simulation

    def run_stationary_test(self):
        simulation = Simulation("stationary_test")
        nb_time_steps = 5
        logging.info(f"Simulating {nb_time_steps} time steps without disruption")
        # print("self.production_capacity", self.firms[0].production_capacity)
        for i in range(nb_time_steps):
            self.run_one_time_step(time_step=i, current_simulation=simulation)
        return simulation

    def save_pickle(self, suffix):
        cache_model(self, suffix)

    def run_criticality_disruption(self, disrupted_edge, duration):
        # Initialize the model
        simulation = Simulation("criticality")
        logging.info("Simulating the initial state")
        self.run_one_time_step(time_step=0, current_simulation=simulation)

        # self.transport_network.compute_flow_per_segment(time_step=0)
        # Get disruptions
        self.disruption_list = DisruptionList([
            TransportDisruption({disrupted_edge: 1.0},
                                Recovery(duration=duration, shape="threshold"))])
        if len(self.disruption_list) == 0:
            raise ValueError("No disruption could be read")
        logging.info(f"{len(self.disruption_list)} disruption(s) will occur")
        self.disruption_list.log_info()

        # Adjust t_final
        t_final = self.parameters.criticality['duration'] + 2
        # t_final = self.parameters.duration_dic[self.disruption_list.end_time]
        logging.info('Simulation will last at max ' + str(t_final) + ' time steps.')

        logging.info("Starting time loop")
        for t in range(1, t_final + 1):
            self.run_one_time_step(time_step=t, current_simulation=simulation)

            if (t > self.disruption_list.end_time + 1) and self.parameters.epsilon_stop_condition:
                if self.is_back_to_equilibrium:
                    logging.info("Simulation stops")
                    break
        return simulation

    def run_disruption(self, t_final: int | None = None):
        # Initialize the model
        simulation = Simulation("event")
        logging.info("Simulating the initial state")
        self.run_one_time_step(time_step=0, current_simulation=simulation)

        # Get disruptions
        self.disruption_list = DisruptionList.from_events_parameter(self.parameters.events,
                                                                    self.parameters.monetary_units_in_model,
                                                                    self.transport_edges, self.firm_table,
                                                                    self.firms)
        if len(self.disruption_list) == 0:
            raise ValueError("No disruption could be read")
        logging.info(f"{len(self.disruption_list)} disruption(s) will occur")
        self.disruption_list.log_info()

        # Adjust t_final
        if not isinstance(t_final, int):
            t_final = self.parameters.duration_dic[self.disruption_list.end_time]
        logging.info('Simulation will last at max ' + str(t_final) + ' time steps.')

        logging.info("Starting time loop")
        for t in range(1, t_final + 1):
            self.run_one_time_step(time_step=t, current_simulation=simulation)

            if (t > max([disruption.start_time for disruption in
                         self.disruption_list])) and self.parameters.epsilon_stop_condition:
                if self.is_back_to_equilibrium:
                    logging.info("Simulation stops")
                    break
        return simulation

    def debug_print(self):
        disrupted_edges = [11992, 11993, 12029, 12033]
        for u, v in self.transport_network.edges:
            edge_data = self.transport_network[u][v]
            if edge_data['id'] in disrupted_edges:
                print(edge_data['id'],
                      sum([shipment['quantity'] for shipment in edge_data['shipments'].values()]))

    def run_one_time_step(self, time_step: int, current_simulation: Simulation):
        logging.info(f"Running time step {time_step}")
        # print("self.production_capacity", self.firms[0].production_capacity)
        # self.transport_network.reset_current_loads(self.parameters.route_optimization_weight)

        available_transport_network = self.transport_network
        if self.disruption_list:
            available_transport_network = self.apply_disruption(time_step)

        self.firms.retrieve_orders(self.sc_network)
        if self.reconstruction_market:
            self.reconstruction_market.evaluate_demand_to_firm(self.firms)
            self.reconstruction_market.send_orders(self.firms)
        # print("self.production_capacity", self.firms[0].production_capacity)
        # print(time_step, "product_stock", self.firms[0].product_stock)
        # print(time_step, "total_order", self.firms[0].total_order)
        self.firms.plan_production(self.sc_network, self.parameters.propagate_input_price_change)
        # print(time_step, "production_target", self.firms[0].production_target)
        self.firms.plan_purchase(self.parameters.adaptive_inventories, self.parameters.adaptive_supplier_weight)
        self.households.send_purchase_orders(self.sc_network)
        self.countries.send_purchase_orders(self.sc_network)
        self.firms.send_purchase_orders(self.sc_network)
        # com = self.sc_network[self.firms[0]][self.households['hh_0']]['object']
        # print(time_step, "com.order", com.order)
        # print(time_step, "production_target", self.firms[0].production_target)
        # print(time_step, "current_production_capacity", self.firms[0].current_production_capacity)
        self.firms.produce()
        # print(time_step, "production", self.firms[0].production)
        # print(time_step, "product_stock", self.firms[0].product_stock
        self.countries.deliver(self.sc_network, self.transport_network, available_transport_network,
                               self.parameters.sectors_no_transport_network,
                               self.parameters.rationing_mode, self.parameters.with_transport,
                               self.parameters.transport_to_households, self.parameters.capacity_constraint,
                               self.parameters.monetary_units_in_model, self.parameters.cost_repercussion_mode,
                               self.parameters.price_increase_threshold, self.parameters.transport_cost_noise_level,
                               self.parameters.use_route_cache)
        self.firms.deliver(self.sc_network, self.transport_network, available_transport_network,
                           self.parameters.sectors_no_transport_network,
                           self.parameters.rationing_mode, self.parameters.with_transport,
                           self.parameters.transport_to_households, self.parameters.capacity_constraint,
                           self.parameters.monetary_units_in_model, self.parameters.cost_repercussion_mode,
                           self.parameters.price_increase_threshold, self.parameters.transport_cost_noise_level,
                           self.parameters.use_route_cache)
        # print(self.firms[0].rationing)
        # print(com.delivery)
        # if time_step == 1:
        #     exit()
        if self.reconstruction_market:
            self.reconstruction_market.distribute_new_capital(self.firms)
        # if congestion: TODO reevaluate modeling of congestion
        #     if (time_step == 0):
        #         transport_network.evaluate_normal_traffic()
        #     else:
        #         transport_network.evaluate_congestion()
        #         if len(transport_network.congestionned_edges) > 0:
        #             logging.info("Nb of congestionned segments: " +
        #                          str(len(transport_network.congestionned_edges)))
        #     for firm in firms:
        #         firm.add_congestion_malus2(sc_network, transport_network)
        #     for country in countries:
        #         country.add_congestion_malus2(sc_network, transport_network)
        #
        if (current_simulation.type not in ['criticality']) and (time_step in [0, 1]):
            current_simulation.transport_network_data += self.transport_network.compute_flow_per_segment(time_step)

        # if (time_step == 0) and (
        # export_sc_flow_analysis):  # should be done at this stage, while the goods are on their way
        #     analyzeSupplyChainFlows(sc_network, firms, export_folder)
        #
        self.households.receive_products(self.sc_network, self.transport_network,
                                         self.parameters.sectors_no_transport_network,
                                         self.parameters.transport_to_households)
        self.countries.receive_products(self.sc_network, self.transport_network,
                                        self.parameters.sectors_no_transport_network)
        self.firms.receive_products(self.sc_network, self.transport_network,
                                    self.parameters.sectors_no_transport_network)
        self.transport_network.check_no_uncollected_shipment()
        self.transport_network.reset_loads()
        self.firms.evaluate_profit(self.sc_network)

        self.transport_network.update_road_disruption_state()
        self.firms.update_disrupted_production_capacity()

        self.store_agent_data(time_step, current_simulation)
        self.store_sc_network_data(time_step, current_simulation)

        compare_production_purchase_plans(self.firms, self.countries, self.households)

    def apply_disruption(self, time_step: int):
        disruptions_starting_now = self.disruption_list.filter_start_time(time_step)
        for disruption in disruptions_starting_now:
            if isinstance(disruption, TransportDisruption):
                disruption.implement(self.transport_network)
            if isinstance(disruption, CapitalDestruction):
                disruption.implement(self.firms, self)
        return self.transport_network.get_undisrupted_network()
        # edge_disruptions_starting_now = disruptions_starting_now.filter_type('transport_edge')
        # if len(edge_disruptions_starting_now) > 0:
        #     self.transport_network.disrupt_edges(
        #         edge_disruptions_starting_now.get_item_id_duration_reduction_dict()
        #     )
        # firm_disruptions_starting_now = disruptions_starting_now.filter_type('firm')
        # if len(firm_disruptions_starting_now) > 0:
        #     self.firms.get_disrupted(firm_disruptions_starting_now.get_item_id_duration_reduction_dict())
        # node disruption not implemented

    @property
    def is_back_to_equilibrium(self):
        household_extra_spending = sum([household.extra_spending for household in self.households.values()])
        household_consumption_loss = sum([household.consumption_loss for household in self.households.values()])
        country_extra_spending = sum([country.extra_spending for country in self.countries.values()])
        country_consumption_loss = sum([country.consumption_loss for country in self.countries.values()])
        if (household_extra_spending <= self.parameters.epsilon_stop_condition) & \
                (household_consumption_loss <= self.parameters.epsilon_stop_condition) & \
                (country_extra_spending <= self.parameters.epsilon_stop_condition) & \
                (country_consumption_loss <= self.parameters.epsilon_stop_condition):
            logging.info('Household and country extra spending and consumption loss are at pre-disruption values.')
            return True
        else:
            return False

    def store_agent_data(self, time_step: int, simulation: Simulation):
        # TODO: could create agent-level method to export stuff
        simulation.firm_data += [
            {
                'time_step': time_step,
                'firm': firm.pid,
                'production': firm.production,
                'profit': firm.profit,
                'transport_cost': firm.finance['costs']['transport'],
                'input_cost': firm.finance['costs']['input'],
                'other_cost': firm.finance['costs']['other'],
                'inventory_duration': firm.current_inventory_duration,
                'generalized_transport_cost': firm.generalized_transport_cost,
                'usd_transported': firm.usd_transported,
                'tons_transported': firm.tons_transported,
                'tonkm_transported': firm.tonkm_transported
            }
            for firm in self.firms.values()
        ]
        simulation.country_data += [
            {
                'time_step': time_step,
                'country': country.pid,
                'generalized_transport_cost': country.generalized_transport_cost,
                'usd_transported': country.usd_transported,
                'tons_transported': country.tons_transported,
                'tonkm_transported': country.tonkm_transported,
                'extra_spending': country.extra_spending,
                'consumption_loss': country.consumption_loss,
                'spending': sum(list(country.qty_purchased.values()))
            }
            for country in self.countries.values()
        ]
        simulation.household_data += [
            {
                'time_step': time_step,
                'household': household.pid,
                'tot_consumption': household.tot_consumption,
                'spending_per_retailer': household.spending_per_retailer,
                'consumption_per_retailer': household.consumption_per_retailer,
                'extra_spending_per_sector': household.extra_spending_per_sector,
                'consumption_loss_per_sector': household.consumption_loss_per_sector,
                'extra_spending': household.extra_spending,
                'consumption_loss': household.consumption_loss
            }
            for household in self.households.values()
        ]

    def store_sc_network_data(self, time_step: int, simulation: Simulation):
        simulation.sc_network_data += [
            {
                'time_step': time_step,
                'pid': link.pid,
                'status': link.status,
                'price': link.price,
                'order': link.order,
                'delivery': link.delivery,
                "fulfilment_rate": link.fulfilment_rate
            }
            for link in list(nx.get_edge_attributes(self.sc_network, "object").values())
            if link.status != "ok"
        ]

    def export_transport_nodes_edges(self):
        self.transport_nodes[['geometry', 'geometry_wkt', 'id', 'long', 'lat']].to_file(
            self.parameters.export_folder / 'transport_nodes.geojson',
            driver="GeoJSON", index=False)
        cost_columns = pd.concat([
            pd.DataFrame(nx.get_edge_attributes(self.transport_network, "cost_per_ton_solid_bulk"),
                         index=["cost_per_ton_solid_bulk"]).transpose(),
            pd.DataFrame(nx.get_edge_attributes(self.transport_network, "cost_per_ton_liquid_bulk"),
                         index=["cost_per_ton_liquid_bulk"]).transpose(),
            pd.DataFrame(nx.get_edge_attributes(self.transport_network, "cost_per_ton_container"),
                         index=["cost_per_ton_container"]).transpose(),
            pd.DataFrame(nx.get_edge_attributes(self.transport_network, "id"), index=["id"]).transpose()
        ], axis=1).set_index("id", drop=True)
        self.transport_edges = pd.concat([self.transport_edges, cost_columns], axis=1)
        self.transport_edges.drop(columns=['node_tuple']).to_file(
            self.parameters.export_folder / 'transport_edges.geojson',
            driver="GeoJSON", index=False)

    def export_agent_tables(self):
        firm_table_to_export = self.firm_table.drop(columns=['id'])
        if 'tuple' in self.firm_table.columns:
            firm_table_to_export = firm_table_to_export.drop(columns=['tuple'])
        firm_table_to_export.to_file(self.parameters.export_folder / 'firm_table.geojson', driver="GeoJSON")
        household_table_to_export = self.household_table.drop(columns=['id'])
        if 'tuple' in self.household_table.columns:
            household_table_to_export = household_table_to_export.drop(columns=['tuple'])
        household_table_to_export.to_file(self.parameters.export_folder / 'household_table.geojson', index="GeoJSON")
