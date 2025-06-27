import logging
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from disruptsc.model.basic_functions import rescale_values, calculate_distance_between_agents

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.network.transport_network import TransportNetwork
    from disruptsc.network.commercial_link import CommercialLink

EPSILON = 1e-6


class BaseAgent:
    """
    Simplified base agent class with only core functionality.
    
    This class contains only the essential attributes and methods that all agents need,
    without transport-specific logic or complex business operations.
    """
    
    def __init__(self, agent_type, pid, od_point=0, region=None, name=None, long=None, lat=None):
        """
        Initialize a base agent with core properties.
        
        Args:
            agent_type: Type of agent ('firm', 'country', 'household')
            pid: Unique identifier for the agent
            od_point: Origin/destination point in transport network
            region: Geographic region where agent is located
            name: Human-readable name for the agent
            long: Longitude coordinate
            lat: Latitude coordinate
        """
        self.agent_type = agent_type
        self.pid = pid
        self.od_point = od_point
        self.name = name
        self.long = long
        self.lat = lat
        self.region = region
        self.cost_profile = 0
        self.usd_per_ton = None
        self.region_sector = None

    def id_str(self):
        """
        Return a human-readable identifier string for this agent.
        """
        return f"{self.agent_type} {self.pid} located {self.od_point} in {self.region}".capitalize()

    def reset_indicators(self):
        """
        Reset agent indicators. Override in subclasses.
        """
        pass

    def assign_cost_profile(self, nb_cost_profiles: int):
        """
        Assign a random cost profile to this agent.
        """
        if nb_cost_profiles > 0:
            self.cost_profile = random.randint(0, nb_cost_profiles - 1)

    def receive_shipment_and_pay(self, commercial_link: "CommercialLink", transport_network: "TransportNetwork"):
        """
        Agent looks for shipments in the transport nodes it is located at.
        It takes those which correspond to the commercial link, receives them,
        and pays the corresponding supplier.
        """
        # Look at available shipments
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
            if self.agent_type == 'firm' and hasattr(self, 'inventory'):
                self.inventory[commercial_link.product] += quantity_delivered
        else:
            # If none is available and if there was order, log it
            if (commercial_link.delivery > 0) and (commercial_link.order > 0):
                import logging
                logging.debug(f"{self.id_str()} - no shipment available for commercial link {commercial_link.pid} "
                             f"({commercial_link.delivery}) of {commercial_link.product})")
            quantity_delivered = 0
            price = 1

        self.update_indicator(quantity_delivered, price, commercial_link)

    def receive_products_and_pay(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                                 sectors_no_transport_network: list, transport_to_households: bool = False):
        """
        Receive products from suppliers and make payments.
        """
        # Reset variables
        self.reset_indicators()

        # For each incoming link, receive product and pay
        for supplier, _ in sc_network.in_edges(self):
            commercial_link = sc_network[supplier][self]['object']
            if commercial_link.use_transport_network:
                self.receive_shipment_and_pay(commercial_link, transport_network)
            else:
                self.receive_service_and_pay(commercial_link)

    def receive_service_and_pay(self, commercial_link):
        """
        Receive services (always available, same price).
        """
        quantity_delivered = commercial_link.delivery
        commercial_link.payment = quantity_delivered * commercial_link.price
        # Update indicator
        self.update_indicator(quantity_delivered, commercial_link.price, commercial_link)

    def update_indicator(self, quantity_delivered: float, price: float, commercial_link: "CommercialLink"):
        """
        When receiving products, agents update some internal variables.
        Override in subclasses for specific behavior.
        """
        import logging
        
        # Log if quantity received differs from what it was supposed to be
        if abs(commercial_link.delivery - quantity_delivered) > EPSILON:
            logging.debug(
                f"{self.id_str()} - Quantity delivered by {commercial_link.supplier_id} is {quantity_delivered};"
                f" It was supposed to be {commercial_link.delivery}.")
        commercial_link.calculate_fulfilment_rate()
        commercial_link.update_status()

    def identify_suppliers(self, region_sector: str, firms: "Firms", nb_suppliers_per_input: float,
                           weight_localization: float, import_label: str, transport_network=None):
        """
        Identify suppliers for a given region_sector.
        
        Returns:
            tuple: (supplier_type, supplier_ids, weights, distances)
                supplier_type: 'country' or 'firm'
                supplier_ids: list of supplier IDs
                weights: list of weights for each supplier
                distances: list of distances to each supplier (for firms) or None (for countries)
        """
        if import_label in region_sector:
            return "country", [region_sector.split('_')[0]], [1], None

        # Firm selection using O(1) index lookup instead of O(N) search
        potential_suppliers = firms.get_firms_by_region_sector(region_sector)
        if region_sector == self.region_sector and self.pid in potential_suppliers:
            potential_suppliers = [pid for pid in potential_suppliers if pid != self.pid]  # Remove self if needed

        if not potential_suppliers:
            raise ValueError(f"{self.id_str().capitalize()}: No supplier for {region_sector}")

        # Vectorized computation using NumPy arrays for performance
        num_suppliers = len(potential_suppliers)
        
        # Extract importances as numpy array
        importances = np.array([firms[firm_pid].importance for firm_pid in potential_suppliers], dtype=float)
        
        # Calculate distances vectorized (still need individual calls, but store in array)
        # Phase 3: Pass transport_network for cached od_point distance lookups
        distances_raw = np.array([calculate_distance_between_agents(self, firms[firm_id], transport_network) 
                                 for firm_id in potential_suppliers], dtype=float)
        
        # Vectorized distance rescaling using optimized rescale_values
        distances = np.array(rescale_values(distances_raw), dtype=float)
        
        # Vectorized weighted importance calculation 
        weighted_importance = importances / (distances ** weight_localization)

        supplier_type, selected_ids, weights = "firm", *self._select_ids_and_weight(potential_suppliers,
                                                    self._get_probabilities(weighted_importance.tolist()),
                                                    round(nb_suppliers_per_input))
        
        # Return distances for selected suppliers (using raw distances, not rescaled)
        selected_distances = [distances_raw[potential_suppliers.index(supplier_id)] for supplier_id in selected_ids]
        
        return supplier_type, selected_ids, weights, selected_distances

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
        """
        Convert monetary flow to tons using USD per ton conversion factor.
        """
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


class BaseAgents(dict):
    """
    Collection class for managing multiple agents.
    
    This is a dictionary-like container that provides utility methods
    for working with collections of agents.
    """
    
    def __init__(self, agent_list=None):
        super().__init__()
        if agent_list is not None:
            for agent in agent_list:
                self[agent.pid] = agent

    def __setitem__(self, key, value):
        if not isinstance(value, BaseAgent):
            raise KeyError(f"Value must be a BaseAgent, but got {type(value)} from {value.__class__.__module__}")
        if not hasattr(value, 'pid'):
            raise ValueError("Value must have a 'pid' attribute")
        super().__setitem__(key, value)

    def add(self, agent: BaseAgent):
        """Add an agent to the collection."""
        if not hasattr(agent, 'pid'):
            raise ValueError("Object must have a 'pid' attribute")
        self[agent.pid] = agent

    def __repr__(self):
        return f"BaseAgents({super().__repr__()})"

    def sum(self, property_name):
        """Sum a property across all agents."""
        total = 0
        for agent in self.values():
            if hasattr(agent, property_name):
                total += getattr(agent, property_name)
            else:
                raise AttributeError(f"Agent does not have the property '{property_name}'")
        return total

    def mean(self, property_name):
        """Calculate mean of a property across all agents."""
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
        """Get a property from all agents in specified format."""
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
        """Group agent IDs by a property value."""
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
        """Send purchase orders for all agents."""
        for agent in self.values():
            if hasattr(agent, 'send_purchase_orders'):
                agent.send_purchase_orders(sc_network)

    def receive_products(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                         sectors_no_transport_network: list, transport_to_households: bool = False, 
                         use_vectorized: bool = False):
        """All agents receive products from their suppliers."""
        if use_vectorized and len(self) > 5:  # Use vectorized approach for larger agent collections
            from disruptsc.agents.vectorized_operations import vectorized_updater
            
            logging.info(f"Using vectorized product reception for {len(self)} {self.agents_type}")
            results = vectorized_updater.vectorized_receive_products(
                self, sc_network, transport_network, sectors_no_transport_network
            )
            logging.info(f"Vectorized reception processed {len(results['deliveries'])} deliveries")
            
            # Log cache performance if available
            if results.get('performance_metrics', {}).get('cache_used', False):
                from disruptsc.network.topology_cache import get_topology_cache
                cache_stats = get_topology_cache().get_cache_statistics()
                logging.debug(f"Cache hit rate: {cache_stats['hit_rate']:.1%} ({cache_stats['hit_count']} hits, {cache_stats['miss_count']} misses)")
        else:
            # Fallback to sequential processing
            for agent in tqdm(self.values(), total=len(self), desc=f"{self.agents_type.capitalize()} receiving products"):
                agent.receive_products_and_pay(sc_network, transport_network, sectors_no_transport_network,
                                               transport_to_households)

    def assign_cost_profile(self, nb_cost_profiles: int):
        """Assign cost profiles to all agents."""
        for agent in self.values():
            agent.assign_cost_profile(nb_cost_profiles)

    def choose_initial_routes(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                              capacity_constraint: bool,
                              explicit_service_firm: bool, transport_to_households: bool,
                              sectors_no_transport_network: list,
                              transport_cost_noise_level: float,
                              monetary_units_in_model: str,
                              parallelized: bool,
                              use_route_cache: bool):
        """
        Choose initial routes for all agents that have transport capabilities.
        
        This method delegates to each agent's choose_initial_routes method if it exists.
        """
        if parallelized and (not capacity_constraint):
            # Parallelized execution when routes are independent
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        agent.choose_initial_routes, sc_network, transport_network, capacity_constraint,
                        explicit_service_firm, transport_to_households, sectors_no_transport_network,
                        transport_cost_noise_level, monetary_units_in_model, use_route_cache
                    )
                    for agent in self.values()
                    if hasattr(agent, 'choose_initial_routes')
                ]
                for future in futures:
                    future.result()
        else:
            # Sequential execution
            for agent in tqdm(self.values(), total=len(self)):
                if hasattr(agent, 'choose_initial_routes'):
                    agent.choose_initial_routes(
                        sc_network, transport_network, capacity_constraint, explicit_service_firm,
                        transport_to_households, sectors_no_transport_network,
                        transport_cost_noise_level, monetary_units_in_model, use_route_cache
                    )

    def deliver(self, sc_network: "ScNetwork", transport_network: "TransportNetwork",
                available_transport_network: "TransportNetwork",
                sectors_no_transport_network: list, rationing_mode: str, with_transport: bool,
                transport_to_households: bool, capacity_constraint: bool,
                monetary_units_in_model: str, cost_repercussion_mode: str, price_increase_threshold: float,
                transport_cost_noise_level: float, use_route_cache: bool):
        """Deliver products for all agents that have delivery capabilities."""
        for agent in tqdm(self.values(), total=len(self), desc=f"{self.agents_type.capitalize()} delivering"):
            if hasattr(agent, 'deliver_products'):
                agent.deliver_products(
                    sc_network, transport_network, available_transport_network,
                    sectors_no_transport_network=sectors_no_transport_network,
                    rationing_mode=rationing_mode,
                    with_transport=with_transport,
                    transport_to_households=transport_to_households,
                    monetary_units_in_model=monetary_units_in_model,
                    cost_repercussion_mode=cost_repercussion_mode,
                    price_increase_threshold=price_increase_threshold,
                    capacity_constraint=capacity_constraint,
                    transport_cost_noise_level=transport_cost_noise_level,
                    use_route_cache=use_route_cache
                )


def determine_nb_suppliers(nb_suppliers_per_input: float, max_nb_of_suppliers=None):
    """
    Draw 1 or 2 depending on the 'nb_suppliers_per_input' parameters.
    
    Args:
        nb_suppliers_per_input: A float number between 1 and 2
        max_nb_of_suppliers: Maximum value not to exceed
    
    Returns:
        int: Number of suppliers (1 or 2)
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