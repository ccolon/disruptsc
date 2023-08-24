import logging
from collections import UserList

import networkx
import pandas
import pandas as pd

from code.network.transport_network import TransportNetwork


class Agent(object):
    def __init__(self, agent_type, pid, odpoint=0, name=None,
                 long=None, lat=None):
        self.agent_type = agent_type
        self.pid = pid
        self.odpoint = odpoint
        self.name = name
        self.long = long
        self.lat = lat
        self.usd_per_ton = None

    def choose_initial_routes(self, sc_network: networkx.DiGraph, transport_network: TransportNetwork,
                              logistic_modes: str | pandas.DataFrame, account_capacity, monetary_unit_flow):
        for edge in sc_network.out_edges(self):
            if edge[1].pid == -1:  # we do not create route for households
                continue
            elif edge[1].odpoint == -1:  # we do not create route for service firms if explicit_service_firms = False
                continue
            else:
                # Get the id of the origin and destination node
                origin_node = self.odpoint
                destination_node = edge[1].odpoint
                if logistic_modes == "specific":
                    cond_from, cond_to = self.get_transport_cond(edge, logistic_modes)
                    logistic_modes = logistic_modes.loc[cond_from & cond_to, "transport_mode"].iloc[0]
                sc_network[self][edge[1]]['object'].transport_mode = logistic_modes
                # Choose the route and the corresponding mode
                route, selected_mode = self.choose_route(
                    transport_network=transport_network,
                    origin_node=origin_node,
                    destination_node=destination_node,
                    account_capacity=account_capacity,
                    accepted_logistics_modes=logistic_modes
                )
                # print(str(self.pid)+" located "+str(self.odpoint)+": I choose this transport mode "+
                #     str(transport_network.give_route_mode(route))+ " to connect to "+
                #     str(edge[1].pid)+" located "+str(edge[1].odpoint))
                # Store it into commercial link object
                sc_network[self][edge[1]]['object'].store_route_information(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="main",
                    transport_network=transport_network
                )

                if account_capacity:
                    self.update_transport_load(edge, monetary_unit_flow, route, sc_network, transport_network)

    def get_transport_cond(self, edge, transport_modes):
        # Define the type of transport mode to use by looking in the transport_mode table
        if self.agent_type == 'firm':
            cond_from = (transport_modes['from'] == "domestic")
        elif self.agent_type == 'country':
            cond_from = (transport_modes['from'] == self.pid)
        else:
            raise ValueError("'self' must be a Firm or a Country")
        if edge[1].agent_type in ['firm', 'household']:  # see what is the other end
            cond_to = (transport_modes['to'] == "domestic")
        elif edge[1].agent_type == 'country':
            cond_to = (transport_modes['to'] == edge[1].pid)
        else:
            raise ValueError("'edge[1]' must be a Firm or a Country")
            # we have not implemented a "sector" condition
        return cond_from, cond_to

    def update_transport_load(self, edge, monetary_unit_flow, route, sc_network, transport_network):
        # Update the "current load" on the transport network
        # if current_load exceed burden, then add burden to the weight
        new_load_in_usd = sc_network[self][edge[1]]['object'].order
        new_load_in_tons = Agent.transformUSD_to_tons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
        transport_network.update_load_on_route(route, new_load_in_tons)

    def choose_route(self, transport_network: TransportNetwork, origin_node: int, destination_node: int,
                     account_capacity: bool, accepted_logistics_modes: str | list):
        """
        The agent choose the delivery route

        The only way re-implemented (vs. Cambodian version) ist that any mode can be chosen

        Keeping here the comments of the Cambodian version
        If the simple case in which there is only one accepted_logistics_modes
        (as defined by the main parameter logistic_modes)
        then it is simply the shortest_route using the appropriate weigh

        If there are several accepted_logistics_modes, then the agent will investigate different route,
        one per accepted_logistics_mode. They will then pick one, with a certain probability taking into account the
        weight This more complex mode is used when, according to the capacity and cost data, all the exports or
        imports are using one route, whereas in the data, we observe still some flows using another mode of

        transport. So we use this to "force" some flow to take the other routes.
        """
        if account_capacity:
            weight_considered = "capacity_weight"
        else:
            weight_considered = "weight"
        route = transport_network.provide_shortest_route(origin_node,
                                                         destination_node,
                                                         route_weight=weight_considered)
        if route is None:
            raise ValueError(f"Agent {self.pid} - No route found from {origin_node} to {destination_node}")
        else:
            return route, accepted_logistics_modes
        # TODO: check if I want to reimplement this complex route choice procedure
        # if accepted_logistics_modes == "any":
        #     route = transport_network.provide_shortest_route(origin_node,
        #                                                      destination_node,
        #                                                      route_weight="weight")
        #     return route, accepted_logistics_modes
        #
        # else:
        #     logging.error(f'accepted_logistics_modes is {accepted_logistics_modes}')
        #     raise ValueError("The only implemented accepted_logistics_modes is 'any'")
        # # If it is a list, it means that the agent will chosen between different logistic corridors
        # # with a certain probability
        # elif isinstance(accepted_logistics_modes, list):
        #     # pick routes for each modes
        #     routes = {
        #         mode: transport_network.provide_shortest_route(origin_node,
        #                                                        destination_node, route_weight=mode + "_weight")
        #         for mode in accepted_logistics_modes
        #     }
        #     # compute associated weight and capacity_weight
        #     modes_weight = {
        #         mode: {
        #             mode + "_weight": transport_network.sum_indicator_on_route(route, mode + "_weight"),
        #             "weight": transport_network.sum_indicator_on_route(route, "weight"),
        #             "capacity_weight": transport_network.sum_indicator_on_route(route, "capacity_weight")
        #         }
        #         for mode, route in routes.items()
        #     }
        #     # remove any mode which is over capacity (where capacity_weight > capacity_burden)
        #     for mode, route in routes.items():
        #         if mode != "intl_rail":
        #             if transport_network.check_edge_in_route(route, (2610, 2589)):
        #                 print("(2610, 2589) in", mode)
        #     modes_weight = {
        #         mode: weight_dic['weight']
        #         for mode, weight_dic in modes_weight.items()
        #         if weight_dic['capacity_weight'] < capacity_burden
        #     }
        #     if len(modes_weight) == 0:
        #         logging.warning("All transport modes are over capacity, no route selected!")
        #         return None
        #     # and select one route choosing random weighted choice
        #     selection_weights = rescale_values(list(modes_weight.values()), minimum=0, maximum=0.5)
        #     selection_weights = [1 - w for w in selection_weights]
        #     selected_mode = random.choices(
        #         list(modes_weight.keys()),
        #         weights=selection_weights,
        #         k=1
        #     )[0]
        #     # print("Firm "+str(self.pid)+" chooses "+selected_mode+
        #     #     " to serve a client located "+str(destination_node))
        #     route = routes[selected_mode]
        #     return route, selected_mode
        #
        # raise ValueError("The transport_mode attributes of the commerical link\
        #                   does not belong to ('roads', 'intl_multimodes')")

    @staticmethod
    def check_route_availability(commercial_link, transport_network, which_route='main'):
        """
        Look at the main or alternative route
        at check all edges and nodes in the route
        if one is marked as disrupted, then the whole route is marked as disrupted
        """

        if which_route == 'main':
            route_to_check = commercial_link.route
        elif which_route == 'alternative':
            route_to_check = commercial_link.alternative_route
        else:
            raise KeyError('Wrong value for parameter which_route, admissible values are main and alternative')

        res = 'available'
        for route_segment in route_to_check:
            if len(route_segment) == 2:
                if transport_network[route_segment[0]][route_segment[1]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
            if len(route_segment) == 1:
                if transport_network._node[route_segment[0]]['disruption_duration'] > 0:
                    res = 'disrupted'
                    break
        return res

    @staticmethod
    def transformUSD_to_tons(monetary_flow, monetary_unit, usd_per_ton):
        if usd_per_ton == 0:
            return 0
        else:
            # Load monetary units
            monetary_unit_factor = {
                "mUSD": 1e6,
                "kUSD": 1e3,
                "USD": 1
            }
            factor = monetary_unit_factor[monetary_unit]
            return monetary_flow / (usd_per_ton / factor)


class AgentList(UserList):  # TODO: should rather define a dictionary, such that FirmList[a_pid] return the Firm object
    def __init__(self, agent_list: list[Agent]):
        super().__init__(agent for agent in agent_list if isinstance(agent, Agent))

    def send_purchase_orders(self, sc_network: networkx.DiGraph):
        for agent in self:
            agent.send_purchase_orders(sc_network)

    def deliver(self, sc_network: networkx.DiGraph, transport_network: TransportNetwork,
                sectors_no_transport_network: list, rationing_mode: str, account_capacity: bool,
                monetary_units_in_model: str, cost_repercussion_mode: str):
        for agent in self:
            agent.deliver_products(sc_network, transport_network,
                                   sectors_no_transport_network=sectors_no_transport_network,
                                   rationing_mode=rationing_mode,
                                   monetary_units_in_model=monetary_units_in_model,
                                   cost_repercussion_mode=cost_repercussion_mode,
                                   account_capacity=account_capacity)

    def receive_products(self, sc_network: networkx.DiGraph, transport_network: TransportNetwork,
                         sectors_no_transport_network: list):
        for agent in self:
            agent.receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network)
