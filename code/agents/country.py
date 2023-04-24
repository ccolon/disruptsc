import random
from collections import UserList

import pandas as pd
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

    def select_suppliers(self, graph, firm_list, country_list, sector_table, transport_nodes):
        # Select other country as supplier: transit flows
        self.create_transit_links(graph, country_list)

        # Select suppliers
        ## Identify firms from each sectors
        dic_sector_to_firmid = identify_firms_in_each_sector(firm_list)
        share_exporting_firms = sector_table.set_index('sector')['share_exporting_firms'].to_dict()
        ## Identify odpoints which exports (optional)
        export_odpoints = identify_special_transport_nodes(transport_nodes, "export")
        ## Identify sectors to buy from
        present_sectors = list(set(list(dic_sector_to_firmid.keys())))
        sectors_to_buy_from = list(self.qty_purchased.keys())
        present_sectors_to_buy_from = list(set(present_sectors) & set(sectors_to_buy_from))
        ## For each one of these sectors, select suppliers
        supplier_selection_mode = {
            "importance_export": {
                "export_odpoints": export_odpoints,
                "bonus": 10
                # give more weight to firms located in transport node identified as "export points" (e.g., SEZs)
            }
        }
        for sector in present_sectors_to_buy_from:  # only select suppliers from sectors that are present
            # Identify potential suppliers
            potential_supplier_pid = dic_sector_to_firmid[sector]
            # Evaluate how much to select
            nb_selected_suppliers = max(1, round(len(potential_supplier_pid) * share_exporting_firms[sector]))
            if (nb_selected_suppliers > len(potential_supplier_pid)):
                warnings.warn("The number of supplier to select " + str(nb_selected_suppliers)
                              + " is larger than the number of potential supplier" + str(
                    len(potential_supplier_pid)) + " " + str(share_exporting_firms[sector]))
                # Select supplier and weights
            selected_supplier_ids, supplier_weights = determine_suppliers_and_weights(
                potential_supplier_pid,
                nb_selected_suppliers,
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

    # def choose_route(self, transport_network,
    #                  origin_node, destination_node,
    #                  accepted_logistics_modes):
    #     """
    #     The agent choose the delivery route
    #
    #     If the simple case in which there is only one accepted_logistics_modes (as defined by the main parameter logistic_modes)
    #     then it is simply the shortest_route using the appropriate weigth
    #
    #     If there are several accepted_logistics_modes, then the agent will investigate different route, one per
    #     accepted_logistics_mode. They will then pick one, with a certain probability taking into account the weight
    #     This more complex mode is used when, according to the capacity and cost data, all the exports or importzs are using
    #     one route, whereas in the data, we observe still some flows using another mode of tranpsort. So we use this to "force"
    #     some flow to take the other routes.
    #     """
    #     # If accepted_logistics_modes is a string, then simply pick the shortest route of this logistic mode
    #     if isinstance(accepted_logistics_modes, str):
    #         route = transport_network.provide_shortest_route(origin_node,
    #                                                          destination_node,
    #                                                          route_weight=accepted_logistics_modes + "_weight")
    #         return route, accepted_logistics_modes
    #
    #     # If it is a list, it means that the agent will chosen between different logistic corridors
    #     # with a certain probability
    #     elif isinstance(accepted_logistics_modes, list):
    #         # pick routes for each modes
    #         routes = {
    #             mode: transport_network.provide_shortest_route(origin_node,
    #                                                            destination_node, route_weight=mode + "_weight")
    #             for mode in accepted_logistics_modes
    #         }
    #         # compute associated weight and capacity_weight
    #         modes_weight = {
    #             mode: {
    #                 mode + "_weight": transport_network.sum_indicator_on_route(route, mode + "_weight"),
    #                 "weight": transport_network.sum_indicator_on_route(route, "weight", detail_type=False),
    #                 "capacity_weight": transport_network.sum_indicator_on_route(route, "capacity_weight")
    #             }
    #             for mode, route in routes.items()
    #         }
    #         # print(self.pid, modes_weight)
    #         # remove any mode which is over capacity (where capacity_weight > capacity_burden)
    #         capacity_burden = 1e5
    #         for mode, route in routes.items():
    #             if mode != "intl_rail":
    #                 if transport_network.check_edge_in_route(route, (2610, 2589)):
    #                     print("(2610, 2589) in", mode)
    #             # if weight_dic['capacity_weight'] >= capacity_burden:
    #             #     print(mode, "will be eliminated")
    #         # if modes_weight['intl_rail']['capacity_weight'] >= capacity_burden:
    #         #     print("intl_rail", "will be eliminated")
    #         # else:
    #         #     print("intl_rail", "will not be eliminated")
    #
    #         modes_weight = {
    #             mode: weight_dic['weight']
    #             for mode, weight_dic in modes_weight.items()
    #             if weight_dic['capacity_weight'] < capacity_burden
    #         }
    #         if len(modes_weight) == 0:
    #             logging.warning("All transport modes are over capacity, no route selected!")
    #             return None
    #         # and select one route choosing random weighted choice
    #         selection_weights = rescale_values(list(modes_weight.values()), minimum=0, maximum=0.5)
    #         selection_weights = [1 - w for w in selection_weights]
    #         selected_mode = random.choices(
    #             list(modes_weight.keys()),
    #             weights=selection_weights,
    #             k=1
    #         )[0]
    #         route = routes[selected_mode]
    #         # print("Country "+str(self.pid)+" chooses "+selected_mode+
    #         #     " to serve a client located "+str(destination_node))
    #         # print(transport_network.give_route_mode(route))
    #         return route, selected_mode
    #
    #     raise ValueError("The transport_mode attributes of the commerical link\
    #                       does not belong to ('roads', 'intl_multimodes')")

    def deliver_products(self, graph, transport_network, sectors_no_transport_network,
                         rationing_mode, monetary_units_in_model, cost_repercussion_mode, explicit_service_firm):
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
                        cost_repercussion_mode
                    )
            else:
                if (edge[
                    1].odpoint != -1):  # to non service firms, send shipment through transportation network
                    self.send_shipment(
                        graph[self][edge[1]]['object'],
                        transport_network,
                        monetary_units_in_model,
                        cost_repercussion_mode
                    )
                else:  # if it sends to service firms, nothing to do. price is equilibrium price
                    graph[self][edge[1]]['object'].price = graph[self][edge[1]]['object'].eq_price
                    self.qty_sold += graph[self][edge[1]]['object'].delivery

    def send_shipment(self, commercial_link, transport_network,
                      monetary_units_in_model, cost_repercussion_mode):

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
                accepted_logistics_modes=commercial_link.possible_transport_modes
            )
            # We evaluate the cost of this new route
            if route is not None:
                commercial_link.storeRouteInformation(
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

    def add_congestion_malus2(self, graph, transport_network):
        """Congestion cost are perceived costs, felt by firms, but they do not influence prices paid to transporter, hence do not change price
        """
        if len(transport_network.congestionned_edges) > 0:
            # for each client
            for edge in graph.out_edges(self):
                if graph[self][edge[1]]['object'].current_route == 'main':
                    route_to_check = graph[self][edge[1]]['object'].route
                elif graph[self][edge[1]]['object'].current_route == 'alternative':
                    route_to_check = graph[self][edge[1]]['object'].alternative_route
                else:
                    continue
                # check if the route currently used is congestionned
                if len(set(route_to_check) & set(transport_network.congestionned_edges)) > 0:
                    # if it is, we add its cost to the generalized cost model
                    self.generalized_transport_cost += transport_network.giveCongestionCostOfTime(route_to_check)


class CountryList(AgentList):
    pass
    # def __init__(self, country_list: list[Country]):
    #     super().__init__(country for country in country_list if isinstance(country, Country))
