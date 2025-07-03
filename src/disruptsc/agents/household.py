from typing import TYPE_CHECKING

from disruptsc.model.basic_functions import calculate_distance_between_agents, add_or_increment_dict_key

import logging

from disruptsc.agents.base_agent import BaseAgent, BaseAgents
from disruptsc.network.commercial_link import CommercialLink

if TYPE_CHECKING:
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.agents.firm import Firms
    from disruptsc.agents.country import Countries
    from disruptsc.network.transport_network import TransportNetwork


class Household(BaseAgent):

    def __init__(self, pid, od_point, region, name, long, lat, population, sector_consumption, subregion=None, **kwargs):
        super().__init__(
            agent_type="household",
            name=name,
            pid=pid,
            od_point=od_point,
            region=region,
            long=long,
            lat=lat
        )
        # Parameters depending on data
        self.sector_consumption = sector_consumption
        self.population = population
        self.subregion = subregion
        
        # Dynamic subregion system
        self.subregions = {}
        for key, value in kwargs.items():
            if key.startswith('subregion_'):
                level = key[10:]  # Remove 'subregion_' prefix
                self.subregions[level] = value
        # Parameters depending on network
        self.purchase_plan = {}
        self.retailers = {}
        # Variables reset and updated at each time step
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.consumption_per_sector = {}
        self.consumption_loss_per_sector = {}
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.spending_per_sector = {}
        self.extra_spending_per_sector = {}
        # Cumulated variables reset at beginning and updated at each time step
        self.consumption_loss = 0
        self.extra_spending = 0

    def reset_variables(self):
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        self.extra_spending_per_sector = {}
        self.consumption_loss_per_sector = {}

    def reset_indicators(self):
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        self.extra_spending_per_sector = {}
        self.consumption_loss_per_sector = {}

    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        super().update_indicator(quantity_delivered, price, commercial_link)
        self.consumption_per_retailer[commercial_link.supplier_id] = quantity_delivered
        self.tot_consumption += quantity_delivered
        self.spending_per_retailer[commercial_link.supplier_id] = quantity_delivered * price
        self.tot_spending += quantity_delivered * price
        new_extra_spending = quantity_delivered * (price - commercial_link.eq_price)
        self.extra_spending += new_extra_spending
        add_or_increment_dict_key(self.extra_spending_per_sector, commercial_link.product, new_extra_spending)
        new_consumption_loss = (self.purchase_plan[commercial_link.supplier_id] - quantity_delivered) \
                               * commercial_link.eq_price
        self.consumption_loss += new_consumption_loss
        add_or_increment_dict_key(self.consumption_loss_per_sector, commercial_link.product, new_consumption_loss)
    
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

    def initialize_var_on_purchase_plan(self):
        if len(self.purchase_plan) == 0:
            logging.warning("Households initialize variables based on purchase plan, but it is empty.")

        self.consumption_per_retailer = self.purchase_plan
        self.tot_consumption = sum(list(self.purchase_plan.values()))
        self.consumption_loss_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}
        self.spending_per_retailer = self.consumption_per_retailer
        self.tot_spending = self.tot_consumption
        self.extra_spending_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}

    def select_suppliers(self, sc_network: "ScNetwork", firms: "Firms", countries: "Countries",
                         weight_localization: float, nb_suppliers_per_input: int,
                         sector_types_to_shipment_methods: dict, import_label: str, transport_network=None):
        
        # Batch data collection phase - collect all supplier data before network operations
        all_edges_to_add = []  # For batched NetworkX operations
        all_purchase_plan_updates = {}  # For vectorized dictionary updates
        all_retailer_updates = {}  # For vectorized dictionary updates
        all_client_updates = {}  # For vectorized dictionary updates (supplier.clients)
        
        for region_sector, amount in self.sector_consumption.items():
            supplier_type, retailers, retailer_weights, distances = self.identify_suppliers(region_sector, firms,
                                                                                 nb_suppliers_per_input,
                                                                                 weight_localization,
                                                                                 import_label,
                                                                                 transport_network)
            
            # Prepare data for batch operations
            distance_iter = iter(distances) if distances is not None else iter([None] * len(retailers))
            
            for retailer_id, weight in zip(retailers, retailer_weights):
                # Retrieve the appropriate supplier object from the id
                if supplier_type == "country":
                    supplier_object = countries[retailer_id]
                    link_category = 'import_B2C'
                    product_type = "imports"
                    # For countries, we still need to calculate distance since it's not returned
                    distance = calculate_distance_between_agents(self, supplier_object, transport_network)
                else:
                    supplier_object = firms[retailer_id]
                    link_category = 'domestic_B2C'
                    product_type = supplier_object.sector_type
                    # Use the pre-calculated distance from identify_suppliers
                    distance = next(distance_iter)

                # Create commercial link
                commercial_link = CommercialLink(
                    pid=str(retailer_id) + '->' + str(self.pid),
                    product=region_sector,
                    product_type=product_type,
                    category=link_category,
                    origin_node=supplier_object.od_point,
                    destination_node=self.od_point,
                    supplier_id=retailer_id,
                    buyer_id=self.pid
                )
                
                # Configure commercial link
                commercial_link.determine_transportation_mode(sector_types_to_shipment_methods)
                
                # Collect data for batch operations
                all_edges_to_add.append((supplier_object, self, commercial_link, weight))
                all_purchase_plan_updates[retailer_id] = weight * amount
                all_retailer_updates[retailer_id] = {'sector': region_sector, 'weight': weight}
                
                # Prepare supplier client updates (keyed by supplier object for batch update)
                if supplier_object not in all_client_updates:
                    all_client_updates[supplier_object] = {}
                all_client_updates[supplier_object][self.pid] = {
                    'sector': "households", 'share': 0, 'transport_share': 0, "distance": distance
                }
        
        # Batch NetworkX operations - minimize graph operations
        for supplier_object, buyer, commercial_link, weight in all_edges_to_add:
            sc_network.add_edge(supplier_object, buyer, object=commercial_link)
            # Set edge weight immediately after adding edge
            sc_network[supplier_object][buyer]['weight'] = weight
        
        # Vectorized dictionary updates - batch update all dictionaries at once
        self.purchase_plan.update(all_purchase_plan_updates)
        self.retailers.update(all_retailer_updates)
        
        # Batch update supplier client dictionaries
        for supplier_object, client_updates in all_client_updates.items():
            supplier_object.clients.update(client_updates)

    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                logging.warning("Households: No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            commercial_link = graph[edge[0]][self]['object']
            commercial_link.order = quantity_to_buy

    def receive_products_and_pay(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                                 sectors_no_transport_network: list, transport_to_households: bool = False):
        if not transport_to_households:
            self.reset_indicators()
            for supplier, _ in sc_network.in_edges(self):
                self.receive_service_and_pay(sc_network[supplier][self]['object'])

        else:
            super().receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network)


class Households(BaseAgents):
    def __init__(self, agent_list=None):
        super().__init__(agent_list)
        self.agents_type = "households"
