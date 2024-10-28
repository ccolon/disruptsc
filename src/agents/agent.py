import random
from typing import TYPE_CHECKING
import logging

import pandas

if TYPE_CHECKING:
    from src.network.sc_network import ScNetwork
    from src.network.transport_network import TransportNetwork
    from src.network.commercial_link import CommercialLink


EPSILON = 1e-6


class Agent(object):
    def __init__(self, agent_type, pid, od_point=0, name=None,
                 long=None, lat=None):
        self.agent_type = agent_type
        self.pid = pid
        self.od_point = od_point
        self.name = name
        self.long = long
        self.lat = lat
        self.usd_per_ton = None

    def id_str(self):
        return f"{self.agent_type} {self.pid} located {self.od_point}".capitalize()

    def receive_shipment_and_pay(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork"):
        """Firm look for shipments in the transport nodes it is located
        It takes those which correspond to the commercial link
        It receives them, thereby removing them from the transport network
        Then it pays the corresponding supplier along the commecial link
        """
        # Look at available shipment
        available_shipments = transport_network._node[self.od_point]['shipments']
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
            if self.agent_type == 'firm':
                self.inventory[commercial_link.product] += quantity_delivered

        # If none is available and if there was order, log it
        else:
            if (commercial_link.delivery > 0) and (commercial_link.order > 0):
                logging.info(f"{self.id_str()} - no shipment available for commercial link {commercial_link.pid} "
                             f"({commercial_link.delivery}) of {commercial_link.product})")
            quantity_delivered = 0
            price = 1

        self.update_indicator(quantity_delivered, price, commercial_link)

    def receive_products_and_pay(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                                 sectors_no_transport_network: list):
        # reset variable
        self.reset_indicators()

        # for each incoming link, receive product and pay
        # the way differs between service and shipment
        for edge in sc_network.in_edges(self):
            if sc_network[edge[0]][self]['object'].product_type in sectors_no_transport_network:
                self.receive_service_and_pay(sc_network[edge[0]][self]['object'])
            else:
                self.receive_shipment_and_pay(sc_network[edge[0]][self]['object'], transport_network)

    def receive_service_and_pay(self, commercial_link):
        # Always available, same price
        quantity_delivered = commercial_link.delivery
        commercial_link.payment = quantity_delivered * commercial_link.price
        # Update indicator
        self.update_indicator(quantity_delivered, commercial_link.price, commercial_link)

    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        """When receiving product, agents update some internal variables
        """
        # Log if quantity received differs from what it was supposed to be
        if abs(commercial_link.delivery - quantity_delivered) > EPSILON:
            logging.debug(
                f"{self.id_str()} - Quantity delivered by {commercial_link.supplier_id} is {quantity_delivered};"
                f" It was supposed to be {commercial_link.delivery}.")

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              capacity_constraint: bool,
                              transport_cost_noise_level: float, monetary_unit_flow: str):
        for edge in sc_network.out_edges(self):
            if edge[1].od_point == -1:  # we do not create route for service firms if explicit_service_firms = False
                continue
            else:
                # Get the id of the origin and destination node
                origin_node = self.od_point
                destination_node = edge[1].od_point
                # Choose the route and the corresponding mode
                route, selected_mode = self.choose_route(
                    transport_network=transport_network,
                    origin_node=origin_node,
                    destination_node=destination_node,
                    capacity_constraint=capacity_constraint,
                    transport_cost_noise_level=transport_cost_noise_level
                )
                # print(str(self.pid)+" located "+str(self.od_point)+": I choose this transport mode "+
                #     str(transport_network.give_route_mode(route))+ " to connect to "+
                #     str(edge[1].pid)+" located "+str(edge[1].od_point))
                # Store it into commercial link object
                sc_network[self][edge[1]]['object'].store_route_information(
                    route=route,
                    transport_mode=selected_mode,
                    main_or_alternative="main"
                )

                if capacity_constraint:
                    self.update_transport_load(edge, monetary_unit_flow, route, sc_network, transport_network,
                                               capacity_constraint)

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

    def update_transport_load(self, edge, monetary_unit_flow, route, sc_network, transport_network, capacity_constraint):
        # Update the "current load" on the transport network
        # if current_load exceed burden, then add burden to the weight
        new_load_in_usd = sc_network[self][edge[1]]['object'].order
        new_load_in_tons = Agent.transformUSD_to_tons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
        transport_network.update_load_on_route(route, new_load_in_tons, capacity_constraint)

    def choose_route(self, transport_network: "TransportNetwork", origin_node: int, destination_node: int,
                     capacity_constraint: bool, transport_cost_noise_level: float):
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
        if capacity_constraint:
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
        return route

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

    def reset_indicators(self):
        pass


class Agents(dict):
    def __init__(self, agent_list=None):
        super().__init__()
        if agent_list is not None:
            for agent in agent_list:
                self[agent.pid] = agent

    def __setitem__(self, key, value):
        if not isinstance(value, Agent):
            raise KeyError("Value must be an Agent")
        if not hasattr(value, 'pid'):
            raise ValueError("Value must have a 'pid' attribute")
        super().__setitem__(key, value)

    def add(self, agent: Agent):
        if not hasattr(agent, 'pid'):
            raise ValueError("Object must have a 'pid' attribute")
        self[agent.pid] = agent

    def __repr__(self):
        return f"PidDict({super().__repr__()})"

    def sum(self, property_name):
        total = 0
        for agent in self.values():
            if hasattr(agent, property_name):
                total += getattr(agent, property_name)
            else:
                raise AttributeError(f"Agent does not have the property '{property_name}'")
        return total

    def mean(self, property_name):
        total = 0
        count = 0
        for agent in self.values():
            if hasattr(agent, property_name):
                total += getattr(agent, property_name)
                count += 1
            else:
                raise AttributeError(f"Agent does not have the property '{property_name}'")
        if count == 0:
            raise ValueError(f"No agents with the property '{property_name}' found.")
        return total / count

    def get_properties(self, property_name):
        return {pid: getattr(agent, property_name) for pid, agent in self.items()}

    def send_purchase_orders(self, sc_network: "ScNetwork"):
        for agent in self.values():
            agent.send_purchase_orders(sc_network)

    def deliver(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                sectors_no_transport_network: list, rationing_mode: str, capacity_constraint: bool,
                monetary_units_in_model: str, cost_repercussion_mode: str, price_increase_threshold: float,
                transport_cost_noise_level: float):
        for agent in self.values():
            agent.deliver_products(sc_network, transport_network,
                                   sectors_no_transport_network=sectors_no_transport_network,
                                   rationing_mode=rationing_mode, monetary_units_in_model=monetary_units_in_model,
                                   cost_repercussion_mode=cost_repercussion_mode,
                                   price_increase_threshold=price_increase_threshold,
                                   capacity_constraint=capacity_constraint,
                                   transport_cost_noise_level=transport_cost_noise_level)

    def receive_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         sectors_no_transport_network: list):
        for agent in self.values():
            agent.receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network)


def determine_nb_suppliers(nb_suppliers_per_input: float, max_nb_of_suppliers=None):
    """Draw 1 or 2 depending on the 'nb_suppliers_per_input' parameters

    nb_suppliers_per_input is a float number between 1 and 2

    max_nb_of_suppliers: maximum value not to exceed
    """
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
