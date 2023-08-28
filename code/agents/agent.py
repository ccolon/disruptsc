import random
from typing import TYPE_CHECKING
import logging
from collections import UserList

import networkx
import numpy as np
import pandas
import pandas as pd

from code.model.functions import add_or_increment_dict_key, generate_weights, rescale_values, \
    calculate_distance_between_agents

if TYPE_CHECKING:
    from code.network.sc_network import ScNetwork
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

    def id_str(self):
        return f"{self.agent_type} {self.pid} located {self.odpoint}"

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              logistic_modes: str | pandas.DataFrame, account_capacity: bool,
                              transport_cost_noise_level: float, monetary_unit_flow: str):
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
                    transport_cost_noise_level=transport_cost_noise_level,
                    accepted_logistics_modes=logistic_modes
                )
                # print(str(self.pid)+" located "+str(self.odpoint)+": I choose this transport mode "+
                #     str(transport_network.give_route_mode(route))+ " to connect to "+
                #     str(edge[1].pid)+" located "+str(edge[1].odpoint))
                # Store it into commercial link object
                sc_network[self][edge[1]]['object'].store_route_information(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="main"
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

    def choose_route(self, transport_network: "TransportNetwork", origin_node: int, destination_node: int,
                     account_capacity: bool, transport_cost_noise_level: float, accepted_logistics_modes: str | list):
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
                                                         route_weight=weight_considered,
                                                         noise_level=transport_cost_noise_level)
        # if route is None:
        #     raise ValueError(f"Agent {self.pid} - No route found from {origin_node} to {destination_node}")
        # else:
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

    def send_purchase_orders(self, sc_network: "ScNetwork"):
        for agent in self:
            agent.send_purchase_orders(sc_network)

    def deliver(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                sectors_no_transport_network: list, rationing_mode: str, account_capacity: bool,
                monetary_units_in_model: str, cost_repercussion_mode: str, transport_cost_noise_level: float):
        for agent in self:
            agent.deliver_products(sc_network, transport_network,
                                   sectors_no_transport_network=sectors_no_transport_network,
                                   rationing_mode=rationing_mode,
                                   monetary_units_in_model=monetary_units_in_model,
                                   cost_repercussion_mode=cost_repercussion_mode,
                                   account_capacity=account_capacity,
                                   transport_cost_noise_level=transport_cost_noise_level)

    def receive_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         sectors_no_transport_network: list):
        for agent in self:
            agent.receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network)


def determine_nb_suppliers(nb_suppliers_per_input: float, max_nb_of_suppliers=None):
    '''Draw 1 or 2 depending on the 'nb_suppliers_per_input' parameters

    nb_suppliers_per_input is a float number between 1 and 2

    max_nb_of_suppliers: maximum value not to exceed
    '''
    if (nb_suppliers_per_input < 1) or (nb_suppliers_per_input > 2):
        raise ValueError("'nb_suppliers_per_input' should be between 1 and 2")

    if nb_suppliers_per_input == 1:
        nb_suppliers = 1

    elif nb_suppliers_per_input == 2:
        nb_suppliers = 2

    else:
        if random.uniform(0, 1) < nb_suppliers_per_input - 1:
            nb_suppliers = 2
        else:
            nb_suppliers = 1

    if max_nb_of_suppliers:
        nb_suppliers = min(nb_suppliers, max_nb_of_suppliers)

    return nb_suppliers


def select_supplier_from_list(agent, firm_list,
                              nb_suppliers_to_choose, potential_firm_ids,
                              distance, importance, weight_localization,
                              force_same_odpoint=False):
    # reduce firm to choose to local ones
    if force_same_odpoint:
        same_odpoint_firms = [
            firm_id
            for firm_id in potential_firm_ids
            if firm_list[firm_id].odpoint == agent.odpoint
        ]
        if len(same_odpoint_firms) > 0:
            potential_firm_ids = same_odpoint_firms
        #     logging.info('retailer available locally at odpoint '+str(agent.odpoint)+
        #         " for "+firm_list[potential_firm_ids[0]].sector)
        # else:
        #     logging.warning('force_same_odpoint but no retailer available at odpoint '+str(agent.odpoint)+
        #         " for "+firm_list[potential_firm_ids[0]].sector)

    # distance weight
    if distance:
        distance_to_each = rescale_values([
            calculate_distance_between_agents(agent, firm_list[firm_id])
            for firm_id in potential_firm_ids
        ])
        distance_weight = 1 / (np.array(distance_to_each) ** weight_localization)

    # importance weight
    if importance:
        importance_of_each = rescale_values([firm_list[firm_id].importance for firm_id in potential_firm_ids])
        importance_weight = np.array(importance_of_each)

    # create weight vector based on choice
    if importance and distance:
        prob_to_be_selected = importance_weight * distance_weight
    elif importance and not distance:
        prob_to_be_selected = importance_weight
    elif not importance and distance:
        prob_to_be_selected = distance_weight
    else:
        prob_to_be_selected = np.ones((1, len(potential_firm_ids)))
    prob_to_be_selected /= prob_to_be_selected.sum()

    # perform the random choice
    selected_supplier_id = np.random.choice(
        potential_firm_ids,
        p=prob_to_be_selected,
        size=nb_suppliers_to_choose,
        replace=False
    ).tolist()
    # Choose weight if there are multiple suppliers
    if importance:
        supplier_weights = generate_weights(nb_suppliers_to_choose, importance_of_each)
    else:
        supplier_weights = generate_weights(nb_suppliers_to_choose)

    # return
    return selected_supplier_id, supplier_weights


def agent_receive_products_and_pay(agent, graph, transport_network, sectors_no_transport_network):
    # reset variable
    if agent.agent_type == 'country':
        agent.extra_spending = 0
        agent.consumption_loss = 0
    elif agent.agent_type == 'household':
        agent.reset_variables()

    # for each incoming link, receive product and pay
    # the way differs between service and shipment
    for edge in graph.in_edges(agent):
        if graph[edge[0]][agent]['object'].product_type in sectors_no_transport_network:
            agent_receive_service_and_pay(agent, graph[edge[0]][agent]['object'])
        else:
            agent_receive_shipment_and_pay(agent, graph[edge[0]][agent]['object'], transport_network)


def agent_receive_service_and_pay(agent, commercial_link):
    # Always available, same price
    quantity_delivered = commercial_link.delivery
    commercial_link.payment = quantity_delivered * commercial_link.price
    if agent.agent_type == 'firm':
        agent.inventory[commercial_link.product] += quantity_delivered
    # Update indicator
    agent_update_indicator(agent, quantity_delivered, commercial_link.price, commercial_link)


def agent_update_indicator(agent, quantity_delivered, price, commercial_link):
    """When receiving product, agents update some internal variables

    Parameters
    ----------
    """
    if agent.agent_type == "country":
        agent.extra_spending += quantity_delivered * (price - commercial_link.eq_price)
        agent.consumption_loss += commercial_link.delivery - quantity_delivered

    elif agent.agent_type == 'household':
        agent.consumption_per_retailer[commercial_link.supplier_id] = quantity_delivered
        agent.tot_consumption += quantity_delivered
        agent.spending_per_retailer[commercial_link.supplier_id] = quantity_delivered * price
        agent.tot_spending += quantity_delivered * price
        new_extra_spending = quantity_delivered * (price - commercial_link.eq_price)
        agent.extra_spending += new_extra_spending
        add_or_increment_dict_key(agent.extra_spending_per_sector, commercial_link.product, new_extra_spending)
        new_consumption_loss = (agent.purchase_plan[commercial_link.supplier_id] - quantity_delivered) * \
                               commercial_link.eq_price
        agent.consumption_loss += new_consumption_loss
        add_or_increment_dict_key(agent.consumption_loss_per_sector, commercial_link.product, new_consumption_loss)
        # if consum_loss >= 1e-6:
        #     logging.debug("Household "+agent.pid+" Firm "+
        #         str(commercial_link.supplier_id)+" supposed to deliver "+
        #         str(agent.purchase_plan[commercial_link.supplier_id])+
        #         " but delivered "+str(quantity_delivered)
        #     )
    # Log if quantity received differs from what it was supposed to be
    if abs(commercial_link.delivery - quantity_delivered) > 1e-6:
        logging.debug("Agent " + str(agent.pid) + ": quantity delivered by " +
                      str(commercial_link.supplier_id) + " is " + str(quantity_delivered) +
                      ". It was supposed to be " + str(commercial_link.delivery) + ".")


def agent_receive_shipment_and_pay(agent, commercial_link, transport_network):
    """Firm look for shipments in the transport nodes it is located
    It takes those which correspond to the commercial link
    It receives them, thereby removing them from the transport network
    Then it pays the corresponding supplier along the commecial link
    """
    # Look at available shipment
    available_shipments = transport_network._node[agent.odpoint]['shipments']
    if commercial_link.pid in available_shipments.keys():
        # Identify shipment
        shipment = available_shipments[commercial_link.pid]
        # Get quantity and price
        quantity_delivered = shipment['quantity']
        price = shipment['price']
        # Remove shipment from transport
        transport_network.remove_shipment(commercial_link)
        # Make payment
        commercial_link.payment = quantity_delivered * price
        # If firm, add to inventory
        if agent.agent_type == 'firm':
            agent.inventory[commercial_link.product] += quantity_delivered

    # If none is available, log it
    else:
        if commercial_link.delivery > 0:
            logging.info("Agent " + str(agent.pid) +
                         ": no shipment available for commercial link " +
                         str(commercial_link.pid) + ' (' + str(
                commercial_link.delivery) + ' of ' + commercial_link.product + ')'
                         )
        quantity_delivered = 0
        price = 1

    agent_update_indicator(agent, quantity_delivered, price, commercial_link)
