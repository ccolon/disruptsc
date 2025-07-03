from typing import TYPE_CHECKING

import logging
import random

import networkx
import numpy as np
from shapely.geometry import Point

from disruptsc.model.basic_functions import generate_weights, \
    compute_distance_from_arcmin, rescale_values, rescale_monetary_values, generate_weights_from_list

from disruptsc.agents.base_agent import BaseAgent, BaseAgents
from disruptsc.agents.transport_mixin import TransportCapable
from disruptsc.agents.firm_components import (
    ProductionManager, InventoryManager, FinanceManager, SupplierManager,
)
from disruptsc.network.commercial_link import CommercialLink
from disruptsc.parameters import EPSILON

if TYPE_CHECKING:
    from disruptsc.agents.country import Countries
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.network.transport_network import TransportNetwork


class Firm(BaseAgent, TransportCapable):

    def __init__(self, pid, od_point, sector, region_sector, region, sector_type=None, name="noname", input_mix=None,
                 target_margin=0.2, transport_share=0.2, utilization_rate=0.8,
                 importance=1, long=None, lat=None, geometry=None, subregion=None,
                 suppliers=None, clients=None, production=0, min_inventory_duration_target=1,
                 inventory_restoration_time=1,
                 usd_per_ton=2864, capital_to_value_added_ratio=4, **kwargs):
        super().__init__(
            agent_type="firm",
            pid=pid,
            name=name,
            od_point=od_point,
            region=region,
            long=long,
            lat=lat
        )
        # Parameters depending on data
        self.usd_per_ton = usd_per_ton
        self.geometry = geometry
        self.subregion = subregion
        self.importance = importance
        
        # Dynamic subregion system
        self.subregions = {}
        for key, value in kwargs.items():
            if key.startswith('subregion_'):
                level = key[10:]  # Remove 'subregion_' prefix
                self.subregions[level] = value
        self.region_sector = region_sector
        self.sector_type = sector_type
        self.sector = sector
        self.input_mix = input_mix or {}
        
        # Initialize component managers
        self.production_manager = ProductionManager(self)
        self.inventory_manager = InventoryManager(self)
        self.finance_manager = FinanceManager(self)
        self.supplier_manager = SupplierManager(self)
        
        # Configure component managers
        self.finance_manager.target_margin = target_margin
        self.finance_manager.transport_share = transport_share
        self.finance_manager.capital_to_value_added_ratio = capital_to_value_added_ratio
        
        self.production_manager.utilization_rate = utilization_rate
        self.inventory_manager.inventory_duration_target = {input_id: min_inventory_duration_target for input_id in self.input_mix.keys()}
        self.inventory_manager.inventory_restoration_time = inventory_restoration_time
        
        # Equilibrium values
        self.eq_production = 0.0
        self.eq_price = 1.0
        
        # Legacy attributes for backward compatibility
        self.suppliers = suppliers or {}
        self.clients = clients or {}
        self.rationing = 1.0
        
        # Initialize production with given value
        if production > 0:
            self.production_manager.initialize(production, utilization_rate)
        
        # Set up initial equilibrium values from the passed parameters
        self.supplier_manager.eq_total_order = 0.0
        self.eq_production = production

    def reset_variables(self):
        """Reset all variables to their initial state."""
        self.eq_price = 1.0
        self.rationing = 1.0
        
        # Reset component managers
        self.production_manager.reset()
        self.inventory_manager.reset()
        self.finance_manager.reset()
        self.supplier_manager.reset()

    def reset_indicators(self):
        pass

    def id_str(self):
        return super().id_str() + f" sector {self.sector}"
    
    def get_subregion(self, level):
        """Get subregion for specific level (e.g., 'province', 'district')"""
        return self.subregions.get(level)

    def get_all_subregions(self):
        """Get all subregion levels and values"""
        result = {}
        if self.subregion:  # Backward compatibility
            result['default'] = self.subregion
        result.update(self.subregions)
        return result
    
    # Properties for backward compatibility
    @property
    def production(self):
        return self.production_manager.production
    
    @property
    def production_target(self):
        return self.production_manager.production_target
    
    @property
    def product_stock(self):
        return self.production_manager.product_stock
    
    @property
    def inventory(self):
        return self.inventory_manager.inventory
    
    @property
    def purchase_plan(self):
        return self.inventory_manager.purchase_plan
    
    @property
    def order_book(self):
        return self.supplier_manager.order_book
    
    @property
    def total_order(self):
        return self.supplier_manager.total_order
    
    @property
    def profit(self):
        return self.finance_manager.profit
    
    @property
    def finance(self):
        return self.finance_manager.finance
    
    @property
    def delta_price_input(self):
        return self.finance_manager.delta_price_input

    def update_disrupted_production_capacity(self):
        """Update disruption status for production capacity."""
        self.production_manager.update_disrupted_production_capacity()

    def disrupt_production_capacity(self, disruption_duration: int, reduction: float):
        """Apply production capacity disruption."""
        self.production_manager.disrupt_production_capacity(disruption_duration, reduction)

    def initialize_operational_variables(self, eq_production: float, time_resolution: str):
        """Initialize operational variables based on equilibrium production."""
        self.eq_production = eq_production
        
        # Initialize production manager
        self.production_manager.initialize(eq_production, self.production_manager.utilization_rate)
        
        # Initialize inventory manager
        input_needs = {input_pid: self.input_mix[input_pid] * eq_production for input_pid in self.input_mix.keys()}
        min_inventory_duration = min(self.inventory_manager.inventory_duration_target.values()) if self.inventory_manager.inventory_duration_target else 1
        self.inventory_manager.initialize(input_needs, min_inventory_duration)
        
        # Initialize capital
        self.finance_manager.initialize_capital(eq_production, time_resolution)

    def initialize_financial_variables(self, eq_production, eq_input_cost,
                                       eq_transport_cost, eq_other_cost):
        """Initialize financial variables."""
        self.finance_manager.initialize_financial_variables(
            eq_production, eq_input_cost, eq_transport_cost, eq_other_cost
        )
        # Note: equilibrium values are now accessed via properties that delegate to finance_manager

    def add_noise_to_geometry(self, noise_level=1e-5):
        self.geometry = Point(self.long + noise_level * random.uniform(0, 1),
                              self.lat + noise_level * random.uniform(0, 1))

    def distance_to_other(self, other_firm):
        if (self.od_point == -1) or (other_firm.od_point == -1):  # if virtual firms
            return 1
        else:
            return compute_distance_from_arcmin(self.long, self.lat, other_firm.long, other_firm.lat)

    def select_suppliers_from_data(self, graph, firm_list, inputed_supplier_links, output):

        for inputed_supplier_link in list(inputed_supplier_links.transpose().to_dict().values()):
            # Create an edge_attr in the graph
            supplier_id = inputed_supplier_link['supplier_id']
            product_sector = inputed_supplier_link['product_sector']
            supplier_object = firm_list[supplier_id]
            graph.add_edge(supplier_object, self,
                           object=CommercialLink(
                               pid=str(supplier_id) + "->" + str(self.pid),
                               product=product_sector,
                               product_type=supplier_object.sector_type,
                               essential=inputed_supplier_link['is_essential'],
                               category='domestic_B2B',
                               origin_node=supplier_object.od_point,
                               destination_node=self.od_point,
                               supplier_id=supplier_id,
                               buyer_id=self.pid)
                           )
            # Associate a weight to the edge_attr
            weight_in_input_mix = inputed_supplier_link['transaction'] / output
            graph[supplier_object][self]['weight'] = weight_in_input_mix
            # The firm saves the name of the supplier, its sector,
            # its weight among firm of the same sector (without I/O technical coefficient)
            total_input_same_sector = inputed_supplier_links.loc[
                inputed_supplier_links['product_sector'] == product_sector, "transaction"].sum()
            weight_among_same_product = inputed_supplier_link['transaction'] / total_input_same_sector
            self.suppliers[supplier_id] = {'sector': product_sector, 'weight': weight_among_same_product,
                                           "satisfaction": 1}

    def identify_suppliers_legacy(self, region_sector: str, firms: "Firms", countries: "Countries",
                                  nb_suppliers_per_input: float, weight_localization: float,
                                  firm_data_type: str, import_code: str):
        if firm_data_type == "mrio":
            if import_code in region_sector:  # case of countries
                supplier_type = "country"
                selected_supplier_ids = [region_sector.split("_")[0]]  # for countries, id is extracted from the name
                supplier_weights = [1]

            else:  # case of firms
                supplier_type = "firm"
                potential_supplier_pids = [pid for pid, firm in firms.items() if firm.region_sector == region_sector]
                if region_sector == self.region_sector:
                    potential_supplier_pids.remove(self.pid)  # remove oneself
                if len(potential_supplier_pids) == 0:
                    raise ValueError(f"Firm {self.id_str()}: there should be one supplier for {region_sector}")
                # Choose based on importance
                prob_to_be_selected = np.array(rescale_values([firms[firm_pid].importance for firm_pid in
                                                               potential_supplier_pids]))
                prob_to_be_selected /= prob_to_be_selected.sum()
                if nb_suppliers_per_input >= len(potential_supplier_pids):
                    selected_supplier_ids = potential_supplier_pids
                    selected_prob = prob_to_be_selected
                else:
                    selected_supplier_ids = np.random.choice(potential_supplier_pids,
                                                             p=prob_to_be_selected, size=1,
                                                             replace=False).tolist()
                    index_map = {supplier_id: position for position, supplier_id in enumerate(potential_supplier_pids)}
                    selected_positions = [index_map[supplier_id] for supplier_id in selected_supplier_ids]
                    selected_prob = [prob_to_be_selected[position] for position in selected_positions]
                supplier_weights = generate_weights(len(selected_prob), selected_prob)

        else:
            if import_code in region_sector:
                supplier_type = "country"
                # Identify countries as suppliers if the corresponding sector does export
                importance_threshold = 1e-6  # TODO remove
                potential_supplier_pid = [
                    pid
                    for pid, country in countries.items()
                    if country.supply_importance > importance_threshold
                ]
                importance_of_each = [
                    country.supply_importance
                    for country in countries.values()
                    if country.supply_importance > importance_threshold
                ]
                prob_to_be_selected = np.array(importance_of_each)
                prob_to_be_selected /= prob_to_be_selected.sum()

            # For the other types of inputs, identify the domestic suppliers, and
            # calculate their probability to be chosen, based on distance and importance
            else:
                supplier_type = "firm"
                potential_supplier_pid = [pid for pid, firm in firms.items() if firm.region_sector == region_sector]
                if region_sector == self.region_sector:
                    potential_supplier_pid.remove(self.pid)  # remove oneself
                if len(potential_supplier_pid) == 0:
                    raise ValueError(f"Firm {self.pid}: no potential supplier for input {region_sector}")
                distance_to_each = rescale_values([
                    self.distance_to_other(firms[firm_pid])
                    for firm_pid in potential_supplier_pid
                ])  # Compute distance to each of them (vol d oiseau)
                importance_of_each = rescale_values([firms[firm_pid].importance for firm_pid in
                                                     potential_supplier_pid])  # Get importance for each of them
                prob_to_be_selected = np.array(importance_of_each) / (np.array(distance_to_each) ** weight_localization)
                prob_to_be_selected /= prob_to_be_selected.sum()

            # Determine the number of supplier(s) to select. 1 or 2.
            if random.uniform(0, 1) < nb_suppliers_per_input - 1:
                nb_suppliers_to_choose = 2
                if nb_suppliers_to_choose > len(potential_supplier_pid):
                    nb_suppliers_to_choose = 1
            else:
                nb_suppliers_to_choose = 1

            # Select the supplier(s). It there is 2 suppliers, then we generate
            # random weight. It determines how much is bought from each supplier.
            selected_supplier_ids = np.random.choice(potential_supplier_pid,
                                                     p=prob_to_be_selected, size=nb_suppliers_to_choose,
                                                     replace=False).tolist()
            index_map = {supplier_id: position for position, supplier_id in enumerate(potential_supplier_pid)}
            selected_positions = [index_map[supplier_id] for supplier_id in selected_supplier_ids]
            selected_prob = [prob_to_be_selected[position] for position in selected_positions]
            supplier_weights = generate_weights(nb_suppliers_to_choose, selected_prob)

        return supplier_type, selected_supplier_ids, supplier_weights

    def select_suppliers(self, graph: "ScNetwork", firms: "Firms", countries: "Countries",
                         nb_suppliers_per_input: float, weight_localization: float,
                         sector_types_to_shipment_methods: dict, import_label: str, transport_network=None):
        """
        The firm selects its suppliers.

        The firm checks its input mix to identify which type of inputs are needed.
        For each type of input, it selects the appropriate number of suppliers.
        Choice of suppliers is random, based on distance to eligible suppliers and
        their importance.

        If imports are needed, the firms select a country as supplier. Choice is
        random, based on the country's importance.

        Parameters
        ----------
        sector_types_to_shipment_methods
        firm_data_type
        graph : networkx.DiGraph
            Supply chain graph
        firms : list of Firms
            Generated by createFirms function
        countries : list of Countries
            Generated by createCountries function
        nb_suppliers_per_input : float between 1 and 2
            Nb of suppliers per type of inputs. If it is a decimal between 1 and 2,
            some firms will have 1 supplier, other 2 suppliers, such that the
            average matches the specified value.
        weight_localization : float
            Give weight to distance when choosing supplier. The larger, the closer
            the suppliers will be selected.
        import_code : string
            Code that identify imports in the input mix.

        Returns
        -------
        int
            0

        """
        for sector_id, sector_weight in self.input_mix.items():

            # If it is imports, identify international suppliers and calculate
            # their probability to be chosen, which is based on importance.
            supplier_type, selected_supplier_ids, supplier_weights, distances = self.identify_suppliers(sector_id, firms,
                                                                                             nb_suppliers_per_input,
                                                                                             weight_localization,
                                                                                             import_label,
                                                                                             transport_network)

            # For each new supplier, create a new CommercialLink in the supply chain network.
            # print(f"{self.id_str()}: for input {sector_id} I selected {len(selected_supplier_ids)} suppliers")
            for supplier_id in selected_supplier_ids:
                # Retrieve the appropriate supplier object from the id
                # If it is a country we get it from the country list
                # If it is a firm we get it from the firm list
                if supplier_type == "country":
                    supplier_object = countries[supplier_id]
                    link_category = 'import'
                    product_type = "imports"
                else:
                    supplier_object = firms[supplier_id]
                    link_category = 'domestic_B2B'
                    product_type = firms[supplier_id].sector_type
                # Create an edge_attr in the graph
                graph.add_edge(supplier_object, self,
                               object=CommercialLink(
                                   pid=str(supplier_id) + "->" + str(self.pid),
                                   product=sector_id,
                                   product_type=product_type,
                                   category=link_category,
                                   origin_node=supplier_object.od_point,
                                   destination_node=self.od_point,
                                   supplier_id=supplier_id,
                                   buyer_id=self.pid)
                               )
                commercial_link = graph[supplier_object][self]['object']
                commercial_link.determine_transportation_mode(sector_types_to_shipment_methods)
                # Associate a weight, which includes the I/O technical coefficient
                supplier_weight = supplier_weights.pop()
                graph[supplier_object][self]['weight'] = sector_weight * supplier_weight
                # The firm saves the name of the supplier, its sector, its weight (without I/O technical coefficient)
                self.suppliers[supplier_id] = {'sector': sector_id, 'weight': supplier_weight, "satisfaction": 1}
                # The supplier saves the name of the client, its sector, and distance to it.
                # The share of sales cannot be calculated now
                distance = self.distance_to_other(supplier_object)
                supplier_object.clients[self.pid] = {'sector': self.sector, 'share': 0, 'transport_share': 0,
                                                     'distance': distance}

    def calculate_client_share_in_sales(self):
        """Calculate each client's share in total sales."""
        self.supplier_manager.calculate_client_share_in_sales()

    def aggregate_orders(self, log_info=False):
        """Aggregate orders from all clients."""
        self.supplier_manager.aggregate_orders(log_info)

    def decide_production_plan(self):
        """Decide production plan based on orders and stock."""
        self.production_manager.decide_production_plan(self.supplier_manager.total_order)

    def calculate_price(self, graph):
        """Calculate price changes due to input cost changes."""
        self.finance_manager.calculate_price(graph)

    def get_input_costs(self, graph):
        """Get theoretical input costs."""
        return self.finance_manager._get_input_costs(graph)

    def evaluate_input_needs(self):
        """Evaluate input needs based on production target."""
        self.inventory_manager.evaluate_input_needs(self.input_mix, self.production_manager.production_target)

    def decide_purchase_plan(self, adaptive_inventories: bool, adapt_weight_based_on_satisfaction: bool):
        """Decide purchase plan based on inventory needs."""
        self.inventory_manager.decide_purchase_plan(
            adaptive_inventories, adapt_weight_based_on_satisfaction, self.suppliers
        )

    def send_purchase_orders(self, sc_network: "ScNetwork"):
        """Send purchase orders to suppliers."""
        for edge in sc_network.in_edges(self):
            supplier_id = edge[0].pid
            input_sector = edge[0].region_sector
            if supplier_id in self.inventory_manager.purchase_plan.keys():
                quantity_to_buy = self.inventory_manager.purchase_plan[supplier_id]
                if quantity_to_buy == 0:
                    logging.debug(f"{self.id_str()} - I am not planning to buy anything from supplier {supplier_id} "
                                  f"of sector {input_sector}. ")
                    if self.inventory_manager.purchase_plan_per_input[input_sector] == 0:
                        logging.debug(f"{self.id_str()} - I am not planning to buy this input at all. "
                                      f"My needs is {self.inventory_manager.input_needs[input_sector]}, "
                                      f"my inventory is {self.inventory_manager.inventory[input_sector]}, "
                                      f"my inventory target is {self.inventory_manager.inventory_duration_target[input_sector]} and "
                                      f"this input counts {self.input_mix[input_sector]} in my input mix.")
                    else:
                        logging.debug(f"But I plan to buy {self.inventory_manager.purchase_plan_per_input[input_sector]} of this input")
            else:
                raise KeyError(f"{self.id_str()} - Supplier {supplier_id} is not in my purchase plan")
            sc_network[edge[0]][self]['object'].order = quantity_to_buy

    def retrieve_orders(self, sc_network: "ScNetwork"):
        """Retrieve orders from clients."""
        for edge in sc_network.out_edges(self):
            quantity_ordered = sc_network[self][edge[1]]['object'].order
            self.supplier_manager.order_book[edge[1].pid] = quantity_ordered

    def add_reconstruction_order_to_order_book(self):
        """Add reconstruction demand to order book."""
        self.supplier_manager.add_reconstruction_order_to_order_book()

    def evaluate_capacity(self):
        """Evaluate current production capacity."""
        self.production_manager.evaluate_capacity(
            self.finance_manager.capital_destroyed, 
            self.finance_manager.capital_initial
        )

    def incur_capital_destruction(self, amount: float):
        """Apply capital destruction."""
        self.finance_manager.incur_capital_destruction(amount)

    def get_spare_production_potential(self):
        """Calculate spare production capacity."""
        return self.production_manager.get_spare_production_potential(
            self.inventory_manager.inventory, self.input_mix, self.supplier_manager.total_order
        )

    def produce(self, mode="Leontief"):
        """Execute production and update inventories."""
        updated_inventory = self.production_manager.produce(
            self.inventory_manager.inventory, self.input_mix, mode
        )
        self.inventory_manager.inventory = updated_inventory

    def calculate_input_induced_price_change(self, graph):
        """Calculate input-induced price change (legacy method)."""
        eq_theoretical_input_cost, current_theoretical_input_cost = self.get_input_costs(graph)
        input_cost_share = eq_theoretical_input_cost / 1
        relative_change = (current_theoretical_input_cost - eq_theoretical_input_cost) / eq_theoretical_input_cost
        return relative_change * input_cost_share / (1 - self.finance_manager.target_margin)

    def check_if_supplier_changed_price(self, graph):
        """Check if any supplier changed their price."""
        return self.finance_manager._check_if_supplier_changed_price(graph)

    def evaluate_quantities_to_deliver(self, rationing_mode: str):
        """Evaluate quantities to deliver to each client based on stock and orders."""
        # Do nothing if no orders
        if self.supplier_manager.total_order == 0:
            return {buyer_id: 0.0 for buyer_id in self.supplier_manager.order_book.keys()}

        # Otherwise compute rationing factor
        self.rationing = self.production_manager.product_stock / self.supplier_manager.total_order
        # Check the case in which the firm has too much product to sale
        # It should not happen, hence a warning
        if self.rationing > 1 + EPSILON:
            logging.debug(f'Firm {self.pid}: I have produced too much. {self.production_manager.product_stock} vs. {self.supplier_manager.total_order}')
            self.rationing = 1
            quantities_to_deliver = {buyer_id: order for buyer_id, order in self.supplier_manager.order_book.items()}
        # If rationing factor is 1, then it delivers what was ordered
        elif abs(self.rationing - 1) < EPSILON:
            quantities_to_deliver = {buyer_id: order for buyer_id, order in self.supplier_manager.order_book.items()}
        # If rationing occurs, then two rationing behavior: equal or household_first
        elif abs(self.rationing) < EPSILON:
            logging.debug(f'Firm {self.pid}: I have no stock of output, I cannot deliver to my clients')
            quantities_to_deliver = {buyer_id: 0.0 for buyer_id in self.supplier_manager.order_book.keys()}
        else:
            logging.debug(f'Firm {self.pid}: I have to ration my clients by {(1 - self.rationing) * 100:.2f}%')
            # If equal, simply apply rationing factor
            if rationing_mode == "equal":
                quantities_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in
                                         self.supplier_manager.order_book.items()}
            else:
                raise ValueError('Wrong rationing_mode chosen')

            # elif rationing_mode == "household_first": TODO: redo, does not work anymore
            #     if -1 not in self.order_book.keys():
            #         quantity_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in
            #                                self.order_book.items()}
            #     elif len(self.order_book.keys()) == 1:  # only households order to this firm
            #         quantity_to_deliver = {-1: self.total_order}
            #     else:
            #         order_households = self.order_book[-1]
            #         if order_households < self.product_stock:
            #             remaining_product_stock = self.product_stock - order_households
            #             if (self.total_order - order_households) <= 0:
            #                 logging.warning("Firm " + str(self.pid) + ': ' + str(self.total_order - order_households))
            #             rationing_for_business = remaining_product_stock / (self.total_order - order_households)
            #             quantity_to_deliver = {buyer_id: order * rationing_for_business for buyer_id, order in
            #                                    self.order_book.items() if buyer_id != -1}
            #             quantity_to_deliver[-1] = order_households
            #         else:
            #             quantity_to_deliver = {buyer_id: 0 for buyer_id, order in self.order_book.items() if
            #                                    buyer_id != -1}
            #             quantity_to_deliver[-1] = self.product_stock
        # remove rationing as attribute
        return quantities_to_deliver

    def deliver_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         available_transport_network: "TransportNetwork",
                         sectors_no_transport_network: list, rationing_mode: str, with_transport: bool,
                         transport_to_households: bool,
                         monetary_units_in_model: str,
                         cost_repercussion_mode: str, price_increase_threshold: float, capacity_constraint: bool,
                         capacity_constraint_mode: str, use_route_cache: bool):

        quantities_to_deliver = self.evaluate_quantities_to_deliver(rationing_mode)

        # We initialize transport costs, it will be updated for each shipment
        self.finance_manager.finance['costs']['transport'] = 0
        # Transport tracking variables (could be moved to component in future)
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

        # For each client, we define the quantity to deliver then send the shipment
        for _, buyer in sc_network.out_edges(self):
            commercial_link = sc_network[self][buyer]['object']
            quantity_to_deliver = quantities_to_deliver[buyer.pid]
            if quantity_to_deliver == 0:
                if commercial_link.order == 0:
                    logging.debug(f"{self.id_str()} - this client did not order: {buyer.id_str()}")
                continue
            commercial_link.delivery = quantity_to_deliver
            commercial_link.delivery_in_tons = self.transformUSD_to_tons(quantity_to_deliver, monetary_units_in_model,
                                                                         self.usd_per_ton)

            # If the client is B2C (applied only we had one single representative agent for all households)
            cases_no_transport = (buyer.pid == -1) or (self.sector_type in sectors_no_transport_network) \
                                 or ((not transport_to_households) and (buyer.agent_type == "household"))
            # If the client is a service firm, we deliver without using transportation infrastructure
            if cases_no_transport or not with_transport:
                self.deliver_without_infrastructure(commercial_link)
            else:
                self.send_shipment(commercial_link, transport_network, available_transport_network,
                                   price_increase_threshold, capacity_constraint, capacity_constraint_mode,
                                   use_route_cache)

        # For reconstruction orders, we register it
        if isinstance(quantities_to_deliver, int):
            print(quantities_to_deliver)
            raise ValueError()
        self.supplier_manager.reconstruction_produced = quantities_to_deliver.get('reconstruction', 0)

    def deliver_without_infrastructure(self, commercial_link):
        """ The firm deliver its products without using transportation infrastructure
        This case applies to service firm and to households
        Note that we still account for transport cost, proportionally to the share of the clients
        Price can be higher than 1, if there are changes in price inputs
        """
        commercial_link.price = commercial_link.eq_price * (1 + self.finance_manager.delta_price_input)
        self.production_manager.product_stock -= commercial_link.delivery
        self.finance_manager.finance['costs']['transport'] += (self.clients[commercial_link.buyer_id]['share'] *
                                               self.finance_manager.eq_finance['costs']['transport'])

    def record_transport_cost(self, client_id, relative_transport_cost_change):
        """Record transport cost changes (legacy method)."""
        self.finance_manager.record_transport_cost(client_id, relative_transport_cost_change, self.clients)

    def receive_service_and_pay(self, commercial_link: "CommercialLink"):
        super().receive_service_and_pay(commercial_link)
        quantity_delivered = commercial_link.delivery
        self.inventory[commercial_link.product] += quantity_delivered

    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        super().update_indicator(quantity_delivered, price, commercial_link)
        self.suppliers[commercial_link.supplier_id]['satisfaction'] = commercial_link.fulfilment_rate

    def evaluate_profit(self, graph):
        """Evaluate profit based on sales and costs."""
        self.finance_manager.evaluate_profit(graph)
    
    # Implementation of TransportCapable interface methods
    def _update_after_shipment(self, commercial_link: "CommercialLink"):
        """Update firm state after sending a shipment."""
        self.production_manager.product_stock -= commercial_link.delivery
    
    def calculate_relative_price_change_transport(self, relative_transport_cost_change):
        """Calculate price change due to transport cost changes."""
        return self.finance_manager.calculate_relative_price_change_transport(relative_transport_cost_change)
    
    def _record_transport_cost(self, client_id, relative_transport_cost_change):
        """Record transport cost changes."""
        self.finance_manager.record_transport_cost(client_id, relative_transport_cost_change, self.clients)
    
    # Additional properties for backward compatibility
    @property
    def current_production_capacity(self):
        return self.production_manager.current_production_capacity
    
    @property
    def eq_finance(self):
        return self.finance_manager.eq_finance
    
    @property
    def eq_profit(self):
        return self.finance_manager.eq_profit
    
    @property
    def target_margin(self):
        return self.finance_manager.target_margin
    
    @property
    def transport_share(self):
        return self.finance_manager.transport_share
    
    @property
    def capital_initial(self):
        return self.finance_manager.capital_initial
    
    @property
    def capital_destroyed(self):
        return self.finance_manager.capital_destroyed
    
    @property
    def production_capacity_reduction(self):
        return self.production_manager.production_capacity_reduction
    
    @property
    def remaining_disrupted_time(self):
        return self.production_manager.remaining_disrupted_time
    
    @property
    def current_inventory_duration(self):
        return self.inventory_manager.current_inventory_duration
    
    @property
    def eq_needs(self):
        return self.inventory_manager.eq_needs
    
    @property
    def eq_total_order(self):
        return self.supplier_manager.eq_total_order
    
    @eq_total_order.setter
    def eq_total_order(self, value):
        self.supplier_manager.eq_total_order = value
    
    @property
    def inventory_duration_target(self):
        return self.inventory_manager.inventory_duration_target
    
    @property
    def inventory_restoration_time(self):
        return self.inventory_manager.inventory_restoration_time
    
    @property
    def input_needs(self):
        return self.inventory_manager.input_needs
    
    @property
    def purchase_plan_per_input(self):
        return self.inventory_manager.purchase_plan_per_input
    
    @property
    def reconstruction_demand(self):
        return self.supplier_manager.reconstruction_demand
    
    @reconstruction_demand.setter
    def reconstruction_demand(self, value):
        self.supplier_manager.reconstruction_demand = value
    
    @property
    def reconstruction_produced(self):
        return self.supplier_manager.reconstruction_produced
    
    @reconstruction_produced.setter
    def reconstruction_produced(self, value):
        self.supplier_manager.reconstruction_produced = value
    
    # Legacy methods that delegate to component managers
    def receive_service_and_pay(self, commercial_link: "CommercialLink"):
        """Receive services and update inventory."""
        super().receive_service_and_pay(commercial_link)
        quantity_delivered = commercial_link.delivery
        self.inventory_manager.add_to_inventory(commercial_link.product, quantity_delivered)
    
    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        """Update indicators when receiving products."""
        super().update_indicator(quantity_delivered, price, commercial_link)
        self.suppliers[commercial_link.supplier_id]['satisfaction'] = commercial_link.fulfilment_rate


class Firms(BaseAgents):
    def __init__(self, agent_list=None):
        super().__init__(agent_list)
        self.agents_type = "firms"
        self._region_sector_index = {}
        self._index_built = False

    def _build_region_sector_index(self):
        """Build index mapping region_sector -> [firm_ids] for O(1) lookups."""
        self._region_sector_index.clear()
        for firm_id, firm in self.items():
            region_sector = firm.region_sector
            if region_sector not in self._region_sector_index:
                self._region_sector_index[region_sector] = []
            self._region_sector_index[region_sector].append(firm_id)
        self._index_built = True
        logging.debug(f"Built region_sector index with {len(self._region_sector_index)} unique region_sectors")

    def get_firms_by_region_sector(self, region_sector: str) -> list:
        """
        Get list of firm IDs for a given region_sector.
        
        Returns:
            list: List of firm IDs that produce in the given region_sector
        """
        if not self._index_built:
            self._build_region_sector_index()
        return self._region_sector_index.get(region_sector, [])

    def __setitem__(self, key, value):
        """Override to invalidate index when firms are added/modified."""
        super().__setitem__(key, value)
        self._index_built = False

    def __delitem__(self, key):
        """Override to invalidate index when firms are removed."""
        super().__delitem__(key)
        self._index_built = False

    def filter_by_sector(self, sector):
        filtered_agents = Firms()
        for agent in self.values():
            if agent.sector == sector:
                filtered_agents[agent.pid] = agent
        return filtered_agents

    def extract_sectors(self):
        present_sectors = list(self.get_properties('sector', output_type="set"))
        present_region_sectors = list(self.get_properties('region_sector', output_type="set"))
        flow_types_to_export = present_sectors + ['domestic_B2C', 'domestic_B2B', 'transit', 'import',
                                                  'import_B2C', 'export', 'total']
        logging.info(f'Firm_list created, size is: {len(self)}')
        logging.info(f'Number of sectors: {len(present_sectors)}')
        logging.info(f'Sectors present are: {present_sectors}')
        logging.info(f'Number of region sectors: {len(present_region_sectors)}')
        return present_sectors, present_region_sectors, flow_types_to_export

    def retrieve_orders(self, sc_network: "ScNetwork"):
        for firm in self.values():
            firm.retrieve_orders(sc_network)

    def plan_production(self, sc_network: "ScNetwork", propagate_input_price_change: bool = True):
        for firm in self.values():
            firm.evaluate_capacity()
            firm.aggregate_orders(log_info=True)
            firm.decide_production_plan()
            if propagate_input_price_change:
                firm.calculate_price(sc_network)

    def plan_purchase(self, adaptive_inventories: bool, adapt_weight_based_on_satisfaction: bool,
                     use_vectorized: bool = True):
        if use_vectorized and len(self) > 10:  # Use vectorized approach for larger firm collections
            from disruptsc.agents.vectorized_operations import vectorized_updater
            
            logging.debug(f"Using vectorized purchase planning for {len(self)} firms")
            
            # First evaluate input needs for all firms
            for firm in self.values():
                firm.evaluate_input_needs()
            
            # Calculate purchase plans in batch
            purchase_plans = vectorized_updater.vectorized_purchase_planning(
                self, adaptive_inventories
            )
            
            # Apply results back to firms
            for firm_id, purchase_plan in purchase_plans.items():
                if firm_id in self:
                    firm = self[firm_id]
                    # Update the firm's purchase plan
                    firm.inventory_manager.purchase_plan = purchase_plan
                    
                    # Calculate purchase plan per input from supplier plans
                    purchase_plan_per_input = {}
                    for supplier_id, quantity in purchase_plan.items():
                        if supplier_id in firm.suppliers:
                            sector = firm.suppliers[supplier_id]['sector']
                            if sector not in purchase_plan_per_input:
                                purchase_plan_per_input[sector] = 0
                            purchase_plan_per_input[sector] += quantity
                    firm.inventory_manager.purchase_plan_per_input = purchase_plan_per_input
            
            logging.debug(f"Vectorized planning processed {len(purchase_plans)} firms")
        else:
            # Fallback to sequential processing
            for firm in self.values():
                firm.evaluate_input_needs()
                firm.decide_purchase_plan(adaptive_inventories, adapt_weight_based_on_satisfaction)  # mode="reactive"

    def produce(self):
        for firm in self.values():
            firm.produce()

    def evaluate_profit(self, sc_network: "ScNetwork"):
        for firm in self.values():
            firm.evaluate_profit(sc_network)

    def update_disrupted_production_capacity(self):
        for firm in self.values():
            firm.update_disrupted_production_capacity()

    def get_disrupted(self, firm_id_duration_reduction_dict: dict):
        for firm in self.values():
            if firm.pid in list(firm_id_duration_reduction_dict.keys()):
                firm.disrupt_production_capacity(
                    firm_id_duration_reduction_dict[firm.pid]['duration'],
                    firm_id_duration_reduction_dict[firm.pid]['reduction']
                )


# These functions are now imported from firm_components
# They remain here for backward compatibility but will be removed in future versions
