import logging
import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.network.transport_network import TransportNetwork
    from disruptsc.network.commercial_link import CommercialLink

EPSILON = 1e-6


class TransportCapable:
    """
    Mixin providing transport capabilities to agents that need to send shipments.
    
    This mixin extracts transport logic from the base Agent class, allowing only
    agents that actually send shipments (Firms, Countries) to inherit this functionality.
    
    Classes using this mixin must implement:
    - _update_after_shipment(commercial_link): Agent-specific behavior after sending
    - calculate_relative_price_change_transport(relative_cost_change): Price adjustment logic
    
    Classes using this mixin should have these attributes:
    - cost_profile: int
    - od_point: int (origin/destination point in transport network)
    - agent_type: str
    """
    
    def choose_route(self, transport_network: "TransportNetwork", origin_node: int, destination_node: int,
                     shipment_method: str, capacity_constraint: bool, transport_cost_noise_level: float):
        """
        The agent chooses the delivery route based on cost profiles and constraints.
        """
        if capacity_constraint:
            weight_considered = "cost_per_ton_with_capacity_" + str(self.cost_profile)
        else:
            weight_considered = "cost_per_ton_" + str(self.cost_profile)

        return transport_network.provide_shortest_route(origin_node, destination_node, shipment_method,
                                                        route_weight=weight_considered,
                                                        noise_level=transport_cost_noise_level)

    def _get_route(self, transport_network, available_transport_network, destination_node,
                   shipment_method, normal_or_disrupted, capacity_constraint, transport_cost_noise_level,
                   use_route_cache: bool):
        """Internal method to get route with caching support."""
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

    def discover_new_route(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork",
                           available_transport_network: "TransportNetwork",
                           account_capacity: bool, transport_cost_noise_level: float, use_route_cache: bool):
        """
        Discover alternative route when main route is unavailable.
        """
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

    def send_shipment(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork",
                      available_transport_network: "TransportNetwork", price_increase_threshold: float,
                      capacity_constraint: bool, transport_cost_noise_level: float, use_route_cache: bool):
        """
        Send shipment using transport network, with fallback to alternative routes.
        
        This is the main method for shipping products from suppliers to buyers.
        It handles route selection, pricing, and fallback logic for disrupted routes.
        """
        if len(commercial_link.route) == 0:
            raise ValueError(f"{self.id_str()} - commercial link {commercial_link.pid} "
                             f"(qty {commercial_link.order:.02f} is not associated to any route, "
                             f"I cannot send any shipment to the client")

        # Try to use main route
        if commercial_link.route.is_usable(transport_network):
            commercial_link.current_route = 'main'
            commercial_link.price = commercial_link.eq_price
            
            # Apply input price changes for firms
            if self.agent_type == "firm" and hasattr(self, 'delta_price_input'):
                commercial_link.price = commercial_link.price * (1 + self.delta_price_input)
            
            transport_network.transport_shipment(commercial_link, capacity_constraint)
            self._update_after_shipment(commercial_link)
            self._record_transport_cost(commercial_link.buyer_id, 0)
            return 0

        # Main route not available - try alternative route
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
            
            # Record transport cost increase
            self._record_transport_cost(commercial_link.buyer_id, relative_transport_cost_change)
            
            # Calculate total price change including input price changes
            total_relative_price_change = relative_price_change_transport
            if self.agent_type == "firm" and hasattr(self, 'delta_price_input'):
                total_relative_price_change = self.delta_price_input + relative_price_change_transport
            
            if np.isnan(relative_price_change_transport):
                raise ValueError(str(relative_price_change_transport))
            
            commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
            transport_network.transport_shipment(commercial_link, capacity_constraint)
            self._update_after_shipment(commercial_link)
            
            logging.debug(f"{self.id_str().capitalize()}: found an alternative route to {commercial_link.buyer_id}, "
                          f"it is costlier by {100 * relative_price_change_transport:.0f}%, price is "
                          f"{commercial_link.price:.4f} instead of {commercial_link.eq_price:.4f}")

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              capacity_constraint: bool, explicit_service_firm: bool, transport_to_households: bool,
                              sectors_no_transport_network: list, transport_cost_noise_level: float,
                              monetary_unit_flow: str, use_route_cache: bool):
        """
        Choose initial routes for all outgoing commercial links.
        """
        for _, client in sc_network.out_edges(self):
            # Skip if this agent type doesn't use transport
            if self.agent_type == "firm" and hasattr(self, 'sector_type'):
                if self.sector_type in sectors_no_transport_network:
                    continue
            
            # Skip service firms if not explicit
            if (not explicit_service_firm) and (client.agent_type == 'firm'):
                if hasattr(client, 'sector_type') and "service" in client.sector_type:
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
                # logging.info(f"{self.id_str()}, {client.pid}, {route.is_edge_in_route('turkmenbashi', transport_network)}")
                self.update_transport_load(client, monetary_unit_flow, route, sc_network, transport_network,
                                           capacity_constraint)

    def update_transport_load(self, client, monetary_unit_flow, route, sc_network, transport_network,
                              capacity_constraint):
        """
        Update the current load on the transport network.
        """
        from disruptsc.agents.base_agent import BaseAgent  # Import here to avoid circular imports
        
        new_load_in_usd = sc_network[self][client]['object'].order
        new_load_in_tons = BaseAgent.transformUSD_to_tons(new_load_in_usd, monetary_unit_flow, self.usd_per_ton)
        transport_network.update_load_on_route(route, new_load_in_tons, capacity_constraint)

    def assign_cost_profile(self, nb_cost_profiles: int):
        """
        Assign a random cost profile to this agent.
        """
        if nb_cost_profiles > 0:
            self.cost_profile = random.randint(0, nb_cost_profiles - 1)

    def get_transport_cond(self, edge, transport_modes):
        """
        Define the type of transport mode to use by looking in the transport_mode table.
        """
        if self.agent_type == 'firm':
            cond_from = (transport_modes['from'] == "domestic")
        elif self.agent_type == 'country':
            cond_from = (transport_modes['from'] == self.pid)
        else:
            raise ValueError("'self' must be a Firm or a Country")
        
        if edge[1].agent_type in ['firm', 'household']:
            cond_to = (transport_modes['to'] == "domestic")
        elif edge[1].agent_type == 'country':
            cond_to = (transport_modes['to'] == edge[1].pid)
        else:
            raise ValueError("'edge_attr[1]' must be a Firm or a Country")
        
        return cond_from, cond_to

    # Template methods - must be implemented by subclasses
    def _update_after_shipment(self, commercial_link: "CommercialLink"):
        """
        Template method for agent-specific behavior after sending a shipment.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _update_after_shipment")

    def calculate_relative_price_change_transport(self, relative_transport_cost_change):
        """
        Template method for calculating price changes due to transport cost changes.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement calculate_relative_price_change_transport")

    def _record_transport_cost(self, client_id, relative_transport_cost_change):
        """
        Template method for recording transport costs.
        Default implementation does nothing - override in subclasses if needed.
        """
        pass

    # Required attributes/methods that using classes must have
    def id_str(self):
        """Must be implemented by the agent class using this mixin."""
        raise NotImplementedError("Classes using TransportCapable must implement id_str()")