import random
from collections import UserList

import pandas as pd
import geopandas as gpd
import math
import logging
import warnings

from .agent_functions import rescale_values, \
    generate_weights_from_list, \
    determine_suppliers_and_weights, \
    identify_firms_in_each_sector, \
    identify_special_transport_nodes, \
    agent_receive_products_and_pay, calculate_distance_between_agents

from code.agents.agent import Agent, AgentList
from code.network.commercial_link import CommercialLink


class Country(Agent):

    def __init__(self, pid=None, qty_sold=None, qty_purchased=None, odpoint=None, long=None, lat=None,
                 purchase_plan=None, transit_from=None, transit_to=None, supply_importance=None,
                 usd_per_ton=None):
        # Intrinsic parameters
        super().__init__(
            agent_type="country",
            pid=pid,
            odpoint=odpoint,
            long=long,
            lat=lat
        )

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

    def create_transit_links(self, graph, country_list):
        for selling_country_pid, quantity in self.transit_from.items():
            selling_country_object = [country for country in country_list if country.pid == selling_country_pid][0]
            graph.add_edge(selling_country_object, self,
                           object=CommercialLink(
                               pid=str(selling_country_pid) + '->' + str(self.pid),
                               product='transit',
                               product_type="transit",  # suppose that transit type are non service, material stuff
                               category="transit",
                               supplier_id=selling_country_pid,
                               buyer_id=self.pid))
            graph[selling_country_object][self]['weight'] = 1
            self.purchase_plan[selling_country_pid] = quantity
            selling_country_object.clients[self.pid] = {'sector': self.pid, 'share': 0, 'transport_share': 0}

    def select_suppliers(self, graph, firm_list, country_list,
                         sector_table: pd.DataFrame, transport_nodes: gpd.GeoDataFrame):
        # Select other country as supplier: transit flows
        self.create_transit_links(graph, country_list)

        # Identify firms from each sector
        dic_sector_to_firm_id = identify_firms_in_each_sector(firm_list)
        share_exporting_firms = sector_table.set_index('sector')['share_exporting_firms'].to_dict()
        # Identify od_points which exports (optional)
        export_od_points = identify_special_transport_nodes(transport_nodes, "export")
        # Identify sectors to buy from
        present_sectors = list(set(list(dic_sector_to_firm_id.keys())))
        sectors_to_buy_from = list(self.qty_purchased.keys())
        present_sectors_to_buy_from = list(set(present_sectors) & set(sectors_to_buy_from))
        # For each one of these sectors, select suppliers
        supplier_selection_mode = {
            "importance_export": {
                "export_od_points": export_od_points,
                "bonus": 10
                # give more weight to firms located in transport node identified as "export points" (e.g., SEZs)
            }
        }
        for sector in present_sectors_to_buy_from:  # only select suppliers from sectors that are present
            # Identify potential suppliers
            potential_supplier_pid = dic_sector_to_firm_id[sector]
            # Evaluate how much to select
            if sector not in share_exporting_firms:  # case of mrio
                nb_suppliers_to_select = 1
            else:  # otherwise we use the % of the sector table to cal
                nb_suppliers_to_select = max(1, round(len(potential_supplier_pid) * share_exporting_firms[sector]))
            if nb_suppliers_to_select > len(potential_supplier_pid):
                logging.warning(f"The number of supplier to select {nb_suppliers_to_select} "
                                f"is larger than the number of potential supplier {len(potential_supplier_pid)} "
                                f"{share_exporting_firms[sector]}")
                # Select supplier and weights
            selected_supplier_ids, supplier_weights = determine_suppliers_and_weights(
                potential_supplier_pid,
                nb_suppliers_to_select,
                firm_list,
                mode=supplier_selection_mode
            )
            # Materialize the link
            for supplier_id in selected_supplier_ids:
                # For each supplier, create an edge in the economic network
                graph.add_edge(firm_list[supplier_id], self,
                               object=CommercialLink(
                                   pid=str(supplier_id) + '->' + str(self.pid),
                                   product=sector,
                                   product_type=firm_list[supplier_id].sector_type,
                                   category="export",
                                   supplier_id=supplier_id,
                                   buyer_id=self.pid))
                # Associate a weight
                weight = supplier_weights.pop(0)
                graph[firm_list[supplier_id]][self]['weight'] = weight
                # Households save the name of the retailer, its sector, its weight, and adds it to its purchase plan
                self.qty_purchased_perfirm[supplier_id] = {
                    'sector': sector,
                    'weight': weight,
                    'amount': self.qty_purchased[sector] * weight
                }
                self.purchase_plan[supplier_id] = self.qty_purchased[sector] * weight
                # The supplier saves the fact that it exports to this country.
                # The share of sales cannot be calculated now, we put 0 for the moment
                distance = calculate_distance_between_agents(self, firm_list[supplier_id])
                firm_list[supplier_id].clients[self.pid] = {
                    'sector': self.pid, 'share': 0, 'transport_share': 0, 'distance': distance
                }

    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print("Country " + self.pid + ": No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

    def deliver_products(self, graph, transport_network, sectors_no_transport_network,
                         rationing_mode, monetary_units_in_model, cost_repercussion_mode, account_capacity):
        """ The quantity to be delivered is the quantity that was ordered (no rationning takes place)
        """
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0
        self.qty_sold = 0
        for edge in graph.out_edges(self):
            if graph[self][edge[1]]['object'].order == 0:
                logging.debug("Agent " + str(self.pid) + ": " +
                              str(graph[self][edge[1]]['object'].buyer_id) + " is my client but did not order")
                continue
            graph[self][edge[1]]['object'].delivery = graph[self][edge[1]]['object'].order
            graph[self][edge[1]]['object'].delivery_in_tons = \
                Country.transformUSD_to_tons(graph[self][edge[1]]['object'].order, monetary_units_in_model,
                                             self.usd_per_ton)

            explicit_service_firm = True
            if explicit_service_firm:
                # If send services, no use of transport network
                if graph[self][edge[1]]['object'].product_type in sectors_no_transport_network:
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery
                # Otherwise, send shipment through transportation network
                else:
                    self.send_shipment(
                        graph[self][edge[1]]['object'],
                        transport_network,
                        monetary_units_in_model,
                        cost_repercussion_mode,
                        account_capacity
                    )
            else:
                if (edge[1].odpoint != -1):  # to non-service firms, send shipment through transportation network
                    self.send_shipment(
                        graph[self][edge[1]]['object'],
                        transport_network,
                        monetary_units_in_model,
                        cost_repercussion_mode,
                        account_capacity
                    )
                else:  # if it sends to service firms, nothing to do. price is equilibrium price
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery

    def send_shipment(self, commercial_link, transport_network,
                      monetary_units_in_model, cost_repercussion_mode, account_capacity):

        if commercial_link.delivery_in_tons == 0:
            print("delivery", commercial_link.delivery)
            print("supplier_id", commercial_link.supplier_id)
            print("buyer_id", commercial_link.buyer_id)

        monetary_unit_factor = {
            "mUSD": 1e6,
            "kUSD": 1e3,
            "USD": 1
        }
        factor = monetary_unit_factor[monetary_units_in_model]
        """Only apply to B2B flows 
        """
        if len(commercial_link.route) == 0:
            raise ValueError("Country " + str(self.pid) +
                             ": commercial link " + str(commercial_link.pid) +
                             " is not associated to any route, I cannot send any shipment to client " +
                             str(commercial_link.pid))

        if self.check_route_availability(commercial_link, transport_network, 'main') == 'available':
            # If the normal route is available, we can send the shipment as usual and pay the usual price
            commercial_link.current_route = 'main'
            commercial_link.price = commercial_link.eq_price
            transport_network.transport_shipment(commercial_link)

            self.generalized_transport_cost += commercial_link.route_time_cost \
                                               + commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons * commercial_link.route_length
            self.qty_sold += commercial_link.delivery
            return 0

        # If there is a disruption, we try the alternative route, if there is any
        if (len(commercial_link.alternative_route) > 0) & \
                (self.check_route_availability(commercial_link, transport_network, 'alternative') == 'available'):
            commercial_link.current_route = 'alternative'
            route = commercial_link.alternative_route
        # Otherwise we have to find a new one
        else:
            origin_node = self.odpoint
            destination_node = commercial_link.route[-1][0]
            route, selected_mode = self.choose_route(
                transport_network=transport_network.get_undisrupted_network(),
                origin_node=origin_node,
                destination_node=destination_node,
                account_capacity=account_capacity,
                accepted_logistics_modes=commercial_link.possible_transport_modes
            )
            # We evaluate the cost of this new route
            if route is not None:
                commercial_link.store_route_information(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="alternative",
                    transport_network=transport_network
                )

        # If the alternative route is available, or if we discovered one, we proceed
        if route is not None:
            commercial_link.current_route = 'alternative'
            # Calculate contribution to generalized transport cost, to usd/tons/tonkms transported
            self.generalized_transport_cost += commercial_link.alternative_route_time_cost \
                                               + commercial_link.delivery_in_tons * commercial_link.alternative_route_cost_per_ton
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons * commercial_link.alternative_route_length
            self.qty_sold += commercial_link.delivery

            if cost_repercussion_mode == "type1":  # relative cost change with actual bill
                # Calculate relative increase in routing cost
                new_transport_bill = commercial_link.delivery_in_tons * commercial_link.alternative_route_cost_per_ton
                normal_transport_bill = commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton
                relative_cost_change = max(new_transport_bill - normal_transport_bill, 0) / normal_transport_bill
                # print(
                #     self.pid,
                #     commercial_link.delivery_in_tons,
                #     commercial_link.route_cost_per_ton,
                #     commercial_link.alternative_route_cost_per_ton,
                #     relative_cost_change
                # )
                # If switched transport mode, add switching cost
                switching_cost = 0.5
                if commercial_link.alternative_route_mode != commercial_link.route_mode:
                    relative_cost_change = relative_cost_change + switching_cost
                # Translate that into an increase in transport costs in the balance sheet
                relative_price_change_transport = 0.2 * relative_cost_change
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            elif cost_repercussion_mode == "type2":  # actual repercussion de la bill
                added_costUSD_per_ton = max(
                    commercial_link.alternative_route_cost_per_ton - commercial_link.route_cost_per_ton, 0)
                added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton / factor)
                added_costmUSD_per_mUSD = added_costUSD_per_mUSD / factor
                commercial_link.price = commercial_link.eq_price + added_costmUSD_per_mUSD
                relative_price_change_transport = commercial_link.price / commercial_link.eq_price - 1

            elif cost_repercussion_mode == "type3":
                # We translate this real cost into transport cost
                relative_cost_change = (
                                               commercial_link.alternative_route_time_cost - commercial_link.route_time_cost) / commercial_link.route_time_cost
                relative_price_change_transport = 0.2 * relative_cost_change
                # With that, we deliver the shipment
                total_relative_price_change = relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            # If there is an alternative route but it is too expensive
            if relative_price_change_transport > 2:
                logging.info("Country " + str(self.pid) + ": found an alternative route to " +
                             str(commercial_link.buyer_id) + ", but it is costlier by " +
                             '{:.2f}'.format(100 * relative_price_change_transport) + "%, price would be " +
                             '{:.4f}'.format(commercial_link.price) + " instead of " +
                             '{:.4f}'.format(commercial_link.eq_price) +
                             ' so I decide not to send it now.'
                             )
                commercial_link.price = commercial_link.eq_price
                commercial_link.current_route = 'none'
                commercial_link.delivery = 0

            # If there is an alternative route which is not too expensive
            else:
                transport_network.transport_shipment(commercial_link)
                logging.info("Country " + str(self.pid) + ": found an alternative route to client " +
                             str(commercial_link.buyer_id) + ", it is costlier by " +
                             '{:.0f}'.format(100 * relative_price_change_transport) + "%, price is " +
                             '{:.4f}'.format(commercial_link.price) + " instead of " +
                             '{:.4f}'.format(commercial_link.eq_price))

        # It there is no alternative route
        else:
            logging.info("Country " + str(self.pid) + ": because of disruption, there is " +
                         "no route between me and client " + str(commercial_link.buyer_id))
            # We do not write how the input price would have changed
            commercial_link.price = commercial_link.eq_price
            # We do not pay the transporter, so we don't increment the transport cost

    def receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network):
        agent_receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network)

    def evaluate_commercial_balance(self, graph):
        exports = sum([graph[self][edge[1]]['object'].payment for edge in graph.out_edges(self)])
        imports = sum([graph[edge[0]][self]['object'].payment for edge in graph.in_edges(self)])
        print("Country " + self.pid + ": imports " + str(imports) + " from Tanzania and export " + str(
            exports) + " to Tanzania")


class CountryList(AgentList):
    pass
