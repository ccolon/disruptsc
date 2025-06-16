"""
DEPRECATED: This module contains legacy Agent and Agents classes.

These classes have been refactored and split into:
- disruptsc.agents.base_agent.BaseAgent and BaseAgents
- disruptsc.agents.transport_mixin.TransportCapable 
- disruptsc.agents.firm_components (for firm-specific functionality)

Use the new classes instead:
- Firm: inherits from BaseAgent + TransportCapable + uses component managers
- Country: inherits from BaseAgent + TransportCapable
- Household: inherits from BaseAgent

This file will be removed in a future version.
"""

import warnings
import copy
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TYPE_CHECKING
import logging

import numpy as np
import pandas
from tqdm import tqdm

from disruptsc.model.basic_functions import rescale_values, calculate_distance_between_agents

# Warn about deprecated usage
warnings.warn(
    "The agent.py module is deprecated. Use base_agent.py, transport_mixin.py, and firm_components.py instead.",
    DeprecationWarning,
    stacklevel=2
)

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.network.transport_network import TransportNetwork
    from disruptsc.network.commercial_link import CommercialLink

EPSILON = 1e-6


class Agent(object):
    def __init__(self, agent_type, pid, od_point=0, region=None, name=None,
                 long=None, lat=None):
        self.region_sector = None
        self.agent_type = agent_type
        self.pid = pid
        self.od_point = od_point
        self.name = name
        self.long = long
        self.lat = lat
        self.usd_per_ton = None
        self.region = region
        self.cost_profile = 0

    def id_str(self):
        return f"{self.agent_type} {self.pid} located {self.od_point} in {self.region}".capitalize()

    def choose_route(self, transport_network: "TransportNetwork", origin_node: int, destination_node: int,
                     shipment_method: str, capacity_constraint: bool, transport_cost_noise_level: float):
        """
        The agent choose the delivery route
        """
        if capacity_constraint:
            weight_considered = "cost_per_ton_with_capacity_" + str(self.cost_profile)
        else:
            weight_considered = "cost_per_ton_" + str(self.cost_profile)

        return transport_network.provide_shortest_route(origin_node, destination_node, shipment_method,
                                                        route_weight=weight_considered,
                                                        noise_level=transport_cost_noise_level)

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
                logging.debug(f"{self.id_str()} - no shipment available for commercial link {commercial_link.pid} "
                             f"({commercial_link.delivery}) of {commercial_link.product})")
            quantity_delivered = 0
            price = 1

        self.update_indicator(quantity_delivered, price, commercial_link)

    def receive_products_and_pay(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                                 sectors_no_transport_network: list, transport_to_households: bool = False):
        # reset variable
        self.reset_indicators()

        # for each incoming link, receive product and pay
        # the way differs between service and shipment
        for supplier, _ in sc_network.in_edges(self):
            commercial_link = sc_network[supplier][self]['object']
            if commercial_link.use_transport_network:
                self.receive_shipment_and_pay(commercial_link, transport_network)
            else:
                self.receive_service_and_pay(commercial_link)
            # commercial_link.update_status()

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
        commercial_link.calculate_fulfilment_rate()
        commercial_link.update_status()

    def _get_route(self, transport_network, available_transport_network, destination_node,
                   shipment_method, normal_or_disrupted, capacity_constraint, transport_cost_noise_level,
                   use_route_cache: bool):
        # print('_get_route', self.id_str(), destination_node)
        if use_route_cache:
            route = transport_network.retrieve_cached_route(self.od_point, destination_node, self.cost_profile,
                                                            normal_or_disrupted, shipment_method)
            if route:
                return route

        route = self.choose_route(
            transport_network=available_transport_network,
            origin_node=self.od_point,
            destination_node=destination_node,
            shipment_method=shipment_method,
            capacity_constraint=capacity_constraint,
            transport_cost_noise_level=transport_cost_noise_level
        )
        if route and use_route_cache:
            transport_network.cache_route(route, self.od_point, destination_node, self.cost_profile,
                                          normal_or_disrupted, shipment_method)
        return route

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              capacity_constraint: bool, explicit_service_firm: bool, transport_to_households: bool,
                              sectors_no_transport_network: list, transport_cost_noise_level: float,
                              monetary_unit_flow: str, use_route_cache: bool):
        for _, client in sc_network.out_edges(self):
            if self.agent_type == "firm":
                if self.sector_type in sectors_no_transport_network:
                    continue
            if (not explicit_service_firm) and (client.agent_type == 'firm'):
                if "service" in client.sector_type:
                    continue
            elif (not transport_to_households) and (client.agent_type == 'household'):
                continue

            commercial_link = sc_network[self][client]['object']
            destination_node = client.od_point
            route = self._get_route(transport_network, transport_network, destination_node,
                                    commercial_link.shipment_method,
                                    'normal', capacity_constraint, transport_cost_noise_level,
                                    use_route_cache)
            cost_per_ton_label = "cost_per_ton_" + str(self.cost_profile) + "_" + commercial_link.shipment_method
            cost_per_ton = route.sum_indicator(transport_network, cost_per_ton_label)
            commercial_link.store_route_information(route, "main", cost_per_ton)

            if capacity_constraint:
                logging.info(f"{self.id_str()}, {client.pid}, {route.is_edge_in_route('turkmenbashi', transport_network)}")
                self.update_transport_load(client, monetary_unit_flow, route, sc_network, transport_network,
                                           capacity_constraint)

    def discover_new_route(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork",
                           available_transport_network: "TransportNetwork",
                           account_capacity: bool, transport_cost_noise_level: float, use_route_cache: bool):

        destination_node = commercial_link.route[-1][0]
        route = self._get_route(transport_network, available_transport_network, destination_node,
                                commercial_link.shipment_method, 'alternative', account_capacity,
                                transport_cost_noise_level, use_route_cache)

        if route is not None:
            cost_per_ton_label = "cost_per_ton_" + str(self.cost_profile) + "_" + commercial_link.shipment_method
            cost_per_ton = route.sum_indicator(transport_network, cost_per_ton_label)
            commercial_link.store_route_information(route, "alternative", cost_per_ton)
            return True

        return False

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
            raise ValueError("'edge_attr[1]' must be a Firm or a Country")
            # we have not implemented a "sector" condition
        return cond_from, cond_to

    def update_transport_load(self, client, monetary_unit_flow, route, sc_network, transport_network,
                              capacity_constraint):
        # Update the "current load" on the transport network
        # if current_load exceed burden, then add burden to the weight
        new_load_in_usd = sc_network[self][client]['object'].order
        new_load_in_tons = Agent.transformUSD_to_tons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
        transport_network.update_load_on_route(route, new_load_in_tons, capacity_constraint)

    @staticmethod
    def _get_probabilities(weights: list) -> np.ndarray:
        """Normalize and return probability distribution."""
        prob = np.array(weights)
        return prob / prob.sum() if prob.sum() > 0 else np.zeros_like(prob)

    @staticmethod
    def _select_ids_and_weight(potential_suppliers, probabilities, nb_suppliers):
        """
        Select a given number of suppliers from potential suppliers based on their probabilities.
        Returns selected supplier IDs and their corresponding weights.
        """
        selected_supplier_ids = np.random.choice(
            potential_suppliers, size=min(nb_suppliers, len(potential_suppliers)),
            p=probabilities, replace=False
        ).tolist()

        # Normalize weights for selected suppliers
        selected_positions = [potential_suppliers.index(s) for s in selected_supplier_ids]
        selected_weights = [probabilities[pos] for pos in selected_positions]
        selected_weights = np.array(selected_weights) / sum(selected_weights)  # Normalize

        return selected_supplier_ids, selected_weights.tolist()

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

    def identify_suppliers(self, region_sector: str, firms: "Firms", nb_suppliers_per_input: float,
                           weight_localization: float, import_label: str):
        if import_label in region_sector:
            return "country", [region_sector.split('_')[0]], [1]

        # Firm selection
        potential_suppliers = [pid for pid, firm in firms.items() if firm.region_sector == region_sector]
        if region_sector == self.region_sector and self.pid in potential_suppliers:
            potential_suppliers.remove(self.pid)  # Remove self if needed

        if not potential_suppliers:
            raise ValueError(f"{self.id_str().capitalize()}: No supplier for {region_sector}")

        importances = [firms[firm_pid].importance for firm_pid in potential_suppliers]
        distances = rescale_values([calculate_distance_between_agents(self, firms[firm_id])
                                    for firm_id in potential_suppliers])
        weighted_importance = [importances[i] / distances[i] ** weight_localization for i in range(len(importances))]

        return "firm", *self._select_ids_and_weight(potential_suppliers,
                                                    self._get_probabilities(weighted_importance),
                                                    round(nb_suppliers_per_input))

    def send_shipment(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork",
                      available_transport_network: "TransportNetwork", price_increase_threshold: float,
                      capacity_constraint: bool, transport_cost_noise_level: float, use_route_cache: bool):

        if len(commercial_link.route) == 0:
            raise ValueError(f"{self.id_str()} - commercial link {commercial_link.pid} "
                             f"(qty {commercial_link.order:.02} is not associated to any route, "
                             f"I cannot send any shipment to the client")

        if commercial_link.route.is_usable(transport_network):
            commercial_link.current_route = 'main'
            commercial_link.price = commercial_link.eq_price
            if self.agent_type == "firm":
                commercial_link.price = commercial_link.price * (1 + self.delta_price_input)
            transport_network.transport_shipment(commercial_link, capacity_constraint)
            if self.agent_type == "firm":
                self.product_stock -= commercial_link.delivery
                self.record_transport_cost(commercial_link.buyer_id, 0)
            if self.agent_type == "country":
                self.qty_sold += commercial_link.delivery
            return 0

        # If there is an alternative route already discovered, and if it is available,
        # then we use it, otherwise we try to find a new one
        usable_alternative = False
        if commercial_link.alternative_found:
            if commercial_link.alternative_route.is_usable(transport_network):
                usable_alternative = True

        if not usable_alternative:
            usable_alternative = self.discover_new_route(commercial_link, transport_network,
                                                         available_transport_network,
                                                         capacity_constraint, transport_cost_noise_level,
                                                         use_route_cache)

        if usable_alternative:
            relative_transport_cost_change = commercial_link.calculate_relative_increase_in_transport_cost()
            relative_price_change_transport = self.calculate_relative_price_change_transport(
                relative_transport_cost_change)
            if relative_price_change_transport > price_increase_threshold:
                logging.debug(f"{self.id_str()}: found an alternative route to {commercial_link.buyer_id} "
                              f"but it is costlier by {100 * relative_price_change_transport:.2f}%, "
                              f"which exceeds the threshold, so I decide not to send it now.")
                usable_alternative = False

        if not usable_alternative:
            logging.debug(f"{self.id_str()}: because of disruption, there is no usable route between me "
                          f"and agent {commercial_link.buyer_id}")
            commercial_link.price = commercial_link.eq_price
            commercial_link.current_route = 'none'
            commercial_link.delivery = 0

        else:
            commercial_link.current_route = 'alternative'
            # We translate this real cost into transport cost
            if self.agent_type == "firm":
                self.record_transport_cost(commercial_link.buyer_id, relative_transport_cost_change)
            # Calculate the relative price change, including any increase due to the prices of inputs
            total_relative_price_change = relative_price_change_transport
            if self.agent_type == "firm":
                total_relative_price_change = self.delta_price_input + relative_price_change_transport
            if np.isnan(relative_price_change_transport):
                raise ValueError(str(relative_price_change_transport))
            commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
            transport_network.transport_shipment(commercial_link, capacity_constraint)
            if self.agent_type == "firm":
                self.product_stock -= commercial_link.delivery
            # Print information
            logging.debug(f"{self.id_str().capitalize()}: found an alternative route to {commercial_link.buyer_id}, "
                          f"it is costlier by {100 * relative_price_change_transport:.0f}%, price is "
                          f"{commercial_link.price:.4f} instead of {commercial_link.eq_price:.4f}")

            # elif cost_repercussion_mode == "type2":  # actual repercussion de la bill
            #     added_cost_usd_per_ton = max(commercial_link.alternative_route_cost_per_ton -
            #                                  commercial_link.route_cost_per_ton,
            #                                  0)
            #     added_cost_usd_per_musd = added_cost_usd_per_ton / (self.usd_per_ton / factor)
            #     added_cost_usd_per_musd = added_cost_usd_per_musd / factor
            #     added_transport_bill = added_cost_usd_per_musd * commercial_link.delivery
            #     self.finance['costs']['transport'] += \
            #         self.eq_finance['costs']['transport'] + added_transport_bill
            #     commercial_link.price = (commercial_link.eq_price
            #                              + self.delta_price_input
            #                              + added_cost_usd_per_musd)
            #     relative_price_change_transport = \
            #         commercial_link.price / (commercial_link.eq_price + self.delta_price_input) - 1
            #     if (commercial_link.price is None) or (commercial_link.price is np.nan):
            #         raise ValueError("Price should be a float, it is " + str(commercial_link.price))
            #
            #     cost_increase = (commercial_link.alternative_route_cost_per_ton
            #                      - commercial_link.route_cost_per_ton) / commercial_link.route_cost_per_ton
            #
            #     logging.debug(f"Firm {self.pid}"
            #                   f": qty {commercial_link.delivery_in_tons} tons"
            #                   f" increase in route cost per ton {cost_increase}"
            #                   f" increased bill mUSD {added_cost_usd_per_musd * commercial_link.delivery}"
            #                   )
            #
            # elif cost_repercussion_mode == "type3":
            #     relative_cost_change = (commercial_link.alternative_route_time_cost
            #                             - commercial_link.route_time_cost) / commercial_link.route_time_cost
            #     self.finance['costs']['transport'] += (self.eq_finance['costs']['transport']
            #                                            * self.clients[commercial_link.buyer_id]['share']
            #                                            * (1 + relative_cost_change))
            #     relative_price_change_transport = (
            #             self.eq_finance['costs']['transport']
            #             * relative_cost_change
            #             / ((1 - self.target_margin) * self.eq_finance['sales']))
            #
            #     total_relative_price_change = self.delta_price_input + relative_price_change_transport
            #     commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
            # else:
            #     raise NotImplementedError(f"Type {cost_repercussion_mode} not implemented")

    def reset_indicators(self):
        pass

    def assign_cost_profile(self, nb_cost_profiles: int):
        if nb_cost_profiles > 0:
            self.cost_profile = random.randint(0, nb_cost_profiles - 1)


class Agents(dict):
    def __init__(self, agent_list=None):
        super().__init__()
        if agent_list is not None:
            for agent in agent_list:
                self[agent.pid] = agent

    def __setitem__(self, key, value):
        if not isinstance(value, Agent):
            raise KeyError(f"Value must be an Agent, but got {type(value)} from {value.__class__.__module__}")
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

    def get_properties(self, property_name, output_type='dict'):
        if output_type == 'dict':
            return {pid: getattr(agent, property_name) for pid, agent in self.items()}
        elif output_type == 'list':
            return [getattr(agent, property_name) for agent in self.values()]
        elif output_type == 'set':
            return set([getattr(agent, property_name) for agent in self.values()])
        else:
            raise ValueError(f"Output type '{output_type}' not recognized.")

    def select_by_properties(self, filters: dict):
        """
        Select agents where the property values match any of the given values in each filter.
        Example: filters = {'region_sector': [...], 'province': [...]}
        """
        selected = self.values()
        for prop, values in filters.items():
            selected = [agent for agent in selected if getattr(agent, prop, None) in values]
        return self.__class__(selected)

    def group_agent_ids_by_property(self, property_name: str):
        id_to_property_dict = self.get_properties(property_name, output_type='dict')
        property_to_ids_dict = {}
        for id, property in id_to_property_dict.items():
            # Append the id to the list of ids for the current property key
            if property in property_to_ids_dict:
                property_to_ids_dict[property].append(id)
            else:
                property_to_ids_dict[property] = [id]
        return property_to_ids_dict

    def send_purchase_orders(self, sc_network: "ScNetwork"):
        for agent in self.values():
            agent.send_purchase_orders(sc_network)

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              capacity_constraint: bool,
                              explicit_service_firm: bool, transport_to_households: bool,
                              sectors_no_transport_network: list,
                              transport_cost_noise_level: float,
                              monetary_units_in_model: str,
                              parallelized: bool,
                              use_route_cache: bool):
        if parallelized and (
                not capacity_constraint):  # in this case the choice of route is not independent, cannot be parallelized
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(agent.choose_initial_routes, sc_network, transport_network, capacity_constraint,
                                    explicit_service_firm, transport_to_households, sectors_no_transport_network,
                                    transport_cost_noise_level, monetary_units_in_model, use_route_cache)
                    for agent in self.values()
                ]
                for future in futures:
                    future.result()
        else:
            for agent in tqdm(self.values(), total=len(self)):
                agent.choose_initial_routes(sc_network, transport_network, capacity_constraint, explicit_service_firm,
                                            transport_to_households, sectors_no_transport_network,
                                            transport_cost_noise_level, monetary_units_in_model, use_route_cache)

    def deliver(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                available_transport_network: "TransportNetwork",
                sectors_no_transport_network: list, rationing_mode: str, with_transport: bool,
                transport_to_households: bool, capacity_constraint: bool,
                monetary_units_in_model: str, cost_repercussion_mode: str, price_increase_threshold: float,
                transport_cost_noise_level: float, use_route_cache: bool):
        for agent in tqdm(self.values(), total=len(self), desc=f"Delivering"):
            agent.deliver_products(sc_network, transport_network, available_transport_network,
                                   sectors_no_transport_network=sectors_no_transport_network,
                                   rationing_mode=rationing_mode,
                                   with_transport=with_transport,
                                   transport_to_households=transport_to_households,
                                   monetary_units_in_model=monetary_units_in_model,
                                   cost_repercussion_mode=cost_repercussion_mode,
                                   price_increase_threshold=price_increase_threshold,
                                   capacity_constraint=capacity_constraint,
                                   transport_cost_noise_level=transport_cost_noise_level,
                                   use_route_cache=use_route_cache)

    def receive_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         sectors_no_transport_network: list, transport_to_households: bool = False):
        for agent in self.values():
            agent.receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network,
                                           transport_to_households)

    def assign_cost_profile(self, nb_cost_profiles: int):
        for agent in self.values():
            agent.assign_cost_profile(nb_cost_profiles)


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
