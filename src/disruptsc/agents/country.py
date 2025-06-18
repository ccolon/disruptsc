from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import logging

from disruptsc.model.basic_functions import calculate_distance_between_agents, rescale_values, \
    generate_weights_from_list
from disruptsc.agents.base_agent import BaseAgent, BaseAgents
from disruptsc.agents.transport_mixin import TransportCapable
from disruptsc.network.commercial_link import CommercialLink

if TYPE_CHECKING:
    from disruptsc.network.transport_network import TransportNetwork
    from disruptsc.network.sc_network import ScNetwork


class Country(BaseAgent, TransportCapable):

    def __init__(self, pid=None, qty_sold=None, qty_purchased=None, od_point=None, region=None, long=None, lat=None,
                 purchase_plan=None, transit_from=None, transit_to=None, supply_importance=None,
                 usd_per_ton=None, import_label="IMP"):
        # Intrinsic parameters
        super().__init__(
            agent_type="country",
            pid=pid,
            od_point=od_point,
            region=region,
            long=long,
            lat=lat
        )
        self.sector = import_label
        self.region_sector = pid + "_" + import_label  # actually, we could specify the country...

        # Parameter based on data
        self.usd_per_ton = usd_per_ton
        # self.entry_points = entry_points or []
        self.transit_from = transit_from or {}
        self.transit_to = transit_to or {}
        self.supply_importance = supply_importance

        # Parameters depending on supplier-buyer network
        self.clients = {}
        self.purchase_plan = purchase_plan or {}
        self.qty_sold = qty_sold or {}
        self.qty_purchased = qty_purchased or {}
        self.qty_purchased_perfirm = {}

        # Variable
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.extra_spending = 0
        self.consumption_loss = 0

    def reset_variables(self):
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.extra_spending = 0
        self.consumption_loss = 0

    def reset_indicators(self):
        self.extra_spending = 0
        self.consumption_loss = 0

    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        super().update_indicator(quantity_delivered, price, commercial_link)
        self.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        self.consumption_loss += commercial_link.delivery - quantity_delivered

    def create_transit_links(self, graph, countries):
        for selling_country_pid, quantity in self.transit_from.items():
            selling_country_object = [country for pid, country in countries.items() if pid == selling_country_pid][0]
            graph.add_edge(selling_country_object, self,
                           object=CommercialLink(
                               pid=str(selling_country_pid) + '->' + str(self.pid),
                               product='transit',
                               product_type="transit",  # suppose that transit type are non service, material stuff
                               category="transit",
                               origin_node=countries[selling_country_pid].od_point,
                               destination_node=self.od_point,
                               supplier_id=selling_country_pid,
                               buyer_id=self.pid))
            graph[selling_country_object][self]['weight'] = 1
            self.purchase_plan[selling_country_pid] = quantity
            selling_country_object.clients[self.pid] = {'sector': self.pid, 'share': 0, 'transport_share': 0}

    def select_suppliers(self, graph, firms, country_list, sector_table: pd.DataFrame,
                         sector_types_to_shipment_methods: dict):
        # Select other country as supplier: transit flows
        self.create_transit_links(graph, country_list)

        # Identify firms from each sector
        dic_sector_to_firm_id = firms.group_agent_ids_by_property("region_sector")
        share_exporting_firms = sector_table.set_index('sector')['share_exporting_firms'].to_dict()
        # # Identify od_points which exports (optional)
        # if "special" in transport_nodes.columns:  # clean data, make it a transport network method
        #     export_od_points = transport_nodes.dropna(subset=['special'])
        #     export_od_points = export_od_points.loc[export_od_points['special'].str.contains("export"), "id"].tolist()
        #     supplier_selection_mode = {
        #         "importance_export": {
        #             "export_od_points": export_od_points,
        #             "bonus": 10
        #             # give more weight to firms located in transport node identified as "export points" (e.g., SEZs)
        #         }
        # }
        # else:
        supplier_selection_mode = {}
        # Identify sectors to buy from
        present_region_sectors = list(firms.get_properties('region_sector', output_type="set"))
        sectors_to_buy_from = list(self.qty_purchased.keys())
        present_region_sectors_to_buy_from = list(set(present_region_sectors) & set(sectors_to_buy_from))
        # For each one of these sectors, select suppliers
        for region_sectors in present_region_sectors_to_buy_from:  # only select suppliers from sectors that are present
            # Identify potential suppliers
            potential_supplier_pid = dic_sector_to_firm_id[region_sectors]
            # Evaluate how much to select
            if region_sectors not in share_exporting_firms:  # case of mrio
                nb_suppliers_to_select = 1
            else:  # otherwise we use the % of the sector table to cal
                nb_suppliers_to_select = max(1,
                                             round(len(potential_supplier_pid) * share_exporting_firms[region_sectors]))
            if nb_suppliers_to_select > len(potential_supplier_pid):
                logging.warning(f"The number of supplier to select {nb_suppliers_to_select} "
                                f"is larger than the number of potential supplier {len(potential_supplier_pid)} "
                                f"(share_exporting_firms: {share_exporting_firms[region_sectors]})")
                # Select supplier and weights
            selected_supplier_ids, supplier_weights = determine_suppliers_and_weights(
                potential_supplier_pid,
                nb_suppliers_to_select,
                firms,
                mode=supplier_selection_mode
            )
            # Materialize the link
            for supplier_id in selected_supplier_ids:
                # For each supplier, create an edge_attr in the economic network
                product_type = firms[supplier_id].sector_type
                graph.add_edge(firms[supplier_id], self,
                               object=CommercialLink(
                                   pid=str(supplier_id) + '->' + str(self.pid),
                                   product=region_sectors,
                                   product_type=product_type,
                                   category="export",
                                   origin_node=firms[supplier_id].od_point,
                                   destination_node=self.od_point,
                                   supplier_id=supplier_id,
                                   buyer_id=self.pid))
                graph[firms[supplier_id]][self]['object'].determine_transportation_mode(
                    sector_types_to_shipment_methods)
                # Associate a weight
                weight = supplier_weights.pop(0)
                graph[firms[supplier_id]][self]['weight'] = weight
                # Households save the name of the retailer, its sector, its weight, and adds it to its purchase plan
                self.qty_purchased_perfirm[supplier_id] = {
                    'sector': region_sectors,
                    'weight': weight,
                    'amount': self.qty_purchased[region_sectors] * weight
                }
                self.purchase_plan[supplier_id] = self.qty_purchased[region_sectors] * weight
                # The supplier saves the fact that it exports to this country.
                # The share of sales cannot be calculated now, we put 0 for the moment
                distance = calculate_distance_between_agents(self, firms[supplier_id])
                firms[supplier_id].clients[self.pid] = {
                    'sector': self.pid, 'share': 0, 'transport_share': 0, 'distance': distance
                }

    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print(f"Country {self.pid}: No purchase plan for supplier {edge[0].pid}")
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

    def deliver_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         available_transport_network: "TransportNetwork",
                         sectors_no_transport_network: list[str], rationing_mode: str, with_transport: bool,
                         transport_to_households: bool,
                         monetary_units_in_model: str, cost_repercussion_mode: str, price_increase_threshold: float,
                         capacity_constraint: bool, transport_cost_noise_level: float, use_route_cache: bool):
        """ The quantity to be delivered is the quantity that was ordered (no rationing takes place)

        Parameters
        ----------
        explicit_service_firm
        transport_cost_noise_level
        cost_repercussion_mode
        capacity_constraint
        monetary_units_in_model
        sectors_no_transport_network
        transport_network
        sc_network
        price_increase_threshold
        """
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.qty_sold = 0
        for _, buyer in sc_network.out_edges(self):
            commercial_link = sc_network[self][buyer]['object']
            if commercial_link.order == 0:
                logging.debug(f"{self.id_str()} - {buyer.id_str()} is my client but did not order")
                continue
            commercial_link.delivery = commercial_link.order
            commercial_link.delivery_in_tons = \
                self.transformUSD_to_tons(commercial_link.order, monetary_units_in_model,
                                             self.usd_per_ton)

            # if explicit_service_firm:
            # If send services, no use of transport network
            cases_no_transport = (commercial_link.product_type in sectors_no_transport_network) or \
                                 ((not transport_to_households) and (buyer.agent_type == 'household'))
            if cases_no_transport or not with_transport:
                commercial_link.price = commercial_link.eq_price
                self.qty_sold += commercial_link.delivery
            # Otherwise, send shipment through transportation network
            else:
                self.send_shipment(commercial_link, transport_network, available_transport_network,
                                   price_increase_threshold, capacity_constraint, transport_cost_noise_level,
                                   use_route_cache)
            # else:
            #     if buyer.agent_type == 'firm':
            #         if buyer.sector_type == "service":
            #             commercial_link.price = commercial_link.eq_price
            #             self.qty_sold += commercial_link.delivery
            #     else:
            #         self.send_shipment(commercial_link, transport_network, available_transport_network,
            #                            price_increase_threshold, capacity_constraint, transport_cost_noise_level)

    def calculate_relative_price_change_transport(self, relative_transport_cost_change):
        """Calculate price change due to transport cost changes."""
        return 0.2 * relative_transport_cost_change
    
    def _update_after_shipment(self, commercial_link: "CommercialLink"):
        """Update country state after sending a shipment."""
        self.qty_sold += commercial_link.delivery

    def evaluate_commercial_balance(self, graph):
        exports = sum([graph[self][edge[1]]['object'].payment for edge in graph.out_edges(self)])
        imports = sum([graph[edge[0]][self]['object'].payment for edge in graph.in_edges(self)])
        print("Country " + self.pid + ": imports " + str(imports) + " from Tanzania and export " + str(
            exports) + " to Tanzania")


class Countries(BaseAgents):
    pass


def determine_suppliers_and_weights(potential_supplier_pids,
                                    nb_selected_suppliers, firms, mode):
    # Get importance for each of them
    if "importance_export" in mode.keys():
        importance_of_each = rescale_values([
            firms[firm_pid].importance * mode['importance_export']['bonus']
            if firms[firm_pid].od_point in mode['importance_export']['export_od_points']
            else firms[firm_pid].importance
            for firm_pid in potential_supplier_pids
        ])
    else:
        importance_of_each = rescale_values([
            firms[firm_pid].importance
            for firm_pid in potential_supplier_pids
        ])

    # Select supplier
    prob_to_be_selected = np.array(importance_of_each) / np.array(importance_of_each).sum()
    selected_supplier_ids = np.random.choice(potential_supplier_pids,
                                             p=prob_to_be_selected,
                                             size=nb_selected_suppliers,
                                             replace=False
                                             ).tolist()

    # Compute weights, based on importance only
    supplier_weights = generate_weights_from_list([
        firms[firm_pid].importance
        for firm_pid in selected_supplier_ids
    ])

    return selected_supplier_ids, supplier_weights
