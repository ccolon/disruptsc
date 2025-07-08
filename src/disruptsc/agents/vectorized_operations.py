"""
Vectorized operations for agent state updates and inventory calculations.

This module provides high-performance vectorized implementations of common
agent operations that were previously done in loops.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union
import logging
from collections import defaultdict

from disruptsc.model.utils.profiling import profile_function
from disruptsc.network.topology_cache import get_topology_cache


class VectorizedInventoryManager:
    """
    Vectorized inventory calculations for multiple agents simultaneously.
    
    This class processes inventory updates, purchase planning, and stock
    calculations across all agents of a given type in batch operations.
    """
    
    def __init__(self):
        self.epsilon = 1e-6
    
    @profile_function("Vectorized Inventory Update")
    def batch_update_inventories(self, agents: Dict, deliveries: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Update inventories for multiple agents simultaneously.
        
        Parameters
        ----------
        agents : Dict
            Dictionary of agent_id -> agent objects
        deliveries : Dict[str, Dict[str, float]]
            agent_id -> {product_id -> quantity_delivered}
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Updated inventories: agent_id -> {product_id -> new_quantity}
        """
        if not deliveries:
            return {}
        
        # Extract current inventories
        current_inventories = {}
        for agent_id, agent in agents.items():
            if hasattr(agent, 'inventory_manager'):
                current_inventories[agent_id] = agent.inventory_manager.inventory.copy()
            elif hasattr(agent, 'inventory'):
                current_inventories[agent_id] = agent.inventory.copy()
            else:
                current_inventories[agent_id] = {}
        
        # Vectorized inventory updates
        updated_inventories = {}
        for agent_id, delivery_dict in deliveries.items():
            if agent_id not in current_inventories:
                updated_inventories[agent_id] = delivery_dict.copy()
                continue
                
            current_inv = current_inventories[agent_id]
            updated_inv = current_inv.copy()
            
            # Vectorized addition for each product
            for product_id, delivered_qty in delivery_dict.items():
                if product_id in updated_inv:
                    updated_inv[product_id] += delivered_qty
                else:
                    updated_inv[product_id] = delivered_qty
            
            updated_inventories[agent_id] = updated_inv
        
        return updated_inventories
    
    @profile_function("Vectorized Purchase Planning")
    def batch_calculate_purchase_plans(self, agents: Dict, adaptive_inventories: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate purchase plans for multiple agents simultaneously.
        
        Parameters
        ----------
        agents : Dict
            Dictionary of agent_id -> agent objects
        adaptive_inventories : bool
            Whether to use adaptive inventory planning
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Purchase plans: agent_id -> {supplier_id -> quantity}
        """
        purchase_plans = {}
        
        # Collect data for vectorization
        agent_data = []
        for agent_id, agent in agents.items():
            if not hasattr(agent, 'inventory_manager'):
                continue
                
            inv_mgr = agent.inventory_manager
            
            # Choose reference input needs
            ref_needs = inv_mgr.input_needs if adaptive_inventories else inv_mgr.eq_needs
            
            agent_data.append({
                'agent_id': agent_id,
                'inventory': inv_mgr.inventory,
                'input_needs': ref_needs,
                'inventory_targets': inv_mgr.inventory_duration_target,
                'restoration_time': inv_mgr.inventory_restoration_time,
                'suppliers': getattr(agent, 'suppliers', {})
            })
        
        if not agent_data:
            return purchase_plans
        
        # Vectorized calculations
        for data in agent_data:
            agent_id = data['agent_id']
            inventory = data['inventory']
            input_needs = data['input_needs']
            targets = data['inventory_targets']
            restoration_time = data['restoration_time']
            suppliers = data['suppliers']
            
            # Calculate purchase plans per input (vectorized where possible)
            purchase_plan_per_input = {}
            for input_id, need in input_needs.items():
                current_stock = inventory.get(input_id, 0)
                target_duration = targets.get(input_id, 1.0)
                
                # Vectorized purchase planning calculation
                target_stock = need * target_duration
                stock_deficit = max(0, target_stock - current_stock)
                purchase_plan_per_input[input_id] = stock_deficit / restoration_time
            
            # Calculate supplier-specific purchases
            agent_purchase_plan = {}
            for supplier_id, supplier_info in suppliers.items():
                sector = supplier_info.get('sector', '')
                weight = supplier_info.get('weight', 0)
                if sector in purchase_plan_per_input:
                    agent_purchase_plan[supplier_id] = purchase_plan_per_input[sector] * weight
            
            purchase_plans[agent_id] = agent_purchase_plan
        
        return purchase_plans
    
    @profile_function("Vectorized Inventory Duration")
    def batch_calculate_inventory_durations(self, agents: Dict) -> Dict[str, Dict[str, float]]:
        """
        Calculate inventory durations for multiple agents simultaneously.
        
        Parameters
        ----------
        agents : Dict
            Dictionary of agent_id -> agent objects
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Inventory durations: agent_id -> {product_id -> duration_in_days}
        """
        durations = {}
        
        for agent_id, agent in agents.items():
            if not hasattr(agent, 'inventory_manager'):
                continue
                
            inv_mgr = agent.inventory_manager
            agent_durations = {}
            
            for input_id, stock in inv_mgr.inventory.items():
                need = inv_mgr.input_needs.get(input_id, 0)
                if need > self.epsilon:
                    duration = stock / need
                else:
                    duration = float('inf') if stock > 0 else 0
                agent_durations[input_id] = duration
            
            durations[agent_id] = agent_durations
        
        return durations


class VectorizedProductionManager:
    """
    Vectorized production calculations for multiple firms simultaneously.
    """
    
    def __init__(self):
        self.epsilon = 1e-6
    
    @profile_function("Vectorized Production Planning")
    def batch_production_planning(self, firms: Dict) -> Dict[str, float]:
        """
        Calculate production targets for multiple firms simultaneously.
        
        Parameters
        ----------
        firms : Dict
            Dictionary of firm_id -> firm objects
            
        Returns
        -------
        Dict[str, float]
            Production targets: firm_id -> production_target
        """
        production_targets = {}
        
        # Collect data for vectorization
        firm_ids = []
        total_orders = []
        product_stocks = []
        
        for firm_id, firm in firms.items():
            if hasattr(firm, 'production_manager') and hasattr(firm, 'total_order'):
                firm_ids.append(firm_id)
                total_orders.append(getattr(firm, 'total_order', 0))
                product_stocks.append(firm.production_manager.product_stock)
        
        if not firm_ids:
            return production_targets
        
        # Vectorized calculation: production_target = max(0, total_order - product_stock)
        total_orders_array = np.array(total_orders)
        product_stocks_array = np.array(product_stocks)
        production_targets_array = np.maximum(0, total_orders_array - product_stocks_array)
        
        # Map back to firm IDs
        for i, firm_id in enumerate(firm_ids):
            production_targets[firm_id] = production_targets_array[i]
        
        return production_targets
    
    @profile_function("Vectorized Stock Updates")
    def batch_update_product_stocks(self, firms: Dict, deliveries: Dict[str, float]) -> Dict[str, float]:
        """
        Update product stocks for multiple firms after deliveries.
        
        Parameters
        ----------
        firms : Dict
            Dictionary of firm_id -> firm objects
        deliveries : Dict[str, float]
            firm_id -> total_quantity_delivered
            
        Returns
        -------
        Dict[str, float]
            Updated product stocks: firm_id -> new_stock_level
        """
        updated_stocks = {}
        
        for firm_id, delivered_qty in deliveries.items():
            if firm_id in firms and hasattr(firms[firm_id], 'production_manager'):
                current_stock = firms[firm_id].production_manager.product_stock
                updated_stocks[firm_id] = max(0, current_stock - delivered_qty)
            else:
                updated_stocks[firm_id] = 0
        
        return updated_stocks


class VectorizedHouseholdManager:
    """
    Vectorized operations for household agents.
    """
    
    def __init__(self):
        self.epsilon = 1e-6
    
    @profile_function("Vectorized Household Reception")
    def batch_household_reception(self, households: Dict, sc_network, transport_to_households: bool) -> Dict[str, Any]:
        """
        Vectorized implementation of household product reception.
        
        Parameters
        ----------
        households : Dict
            Dictionary of household_id -> household objects
        sc_network : ScNetwork
            Supply chain network
        transport_to_households : bool
            Whether households use transport network
            
        Returns
        -------
        Dict[str, Any]
            Results of vectorized operations
        """
        results = {
            'deliveries': {},
            'payments': {},
            'consumption': {}
        }
        
        if not transport_to_households:
            # Simple service-based delivery for all households
            for household_id, household in households.items():
                household.reset_indicators()
                household_deliveries = {}
                
                for supplier, _ in sc_network.in_edges(household):
                    commercial_link = sc_network[supplier][household]['object']
                    quantity_delivered = commercial_link.delivery
                    commercial_link.payment = quantity_delivered * commercial_link.price
                    
                    # Record delivery
                    household_deliveries[commercial_link.product] = quantity_delivered
                    
                    # Update household indicators
                    household.update_indicator(quantity_delivered, commercial_link.price, commercial_link)
                
                results['deliveries'][household_id] = household_deliveries
        
        return results


class VectorizedCountryManager:
    """
    Vectorized operations for country agents.
    """
    
    def __init__(self):
        self.epsilon = 1e-6
    
    @profile_function("Vectorized Country Operations")
    def batch_country_operations(self, countries: Dict, operation_type: str, **kwargs) -> Dict[str, Any]:
        """
        Vectorized implementation of country operations.
        
        Parameters
        ----------
        countries : Dict
            Dictionary of country_id -> country objects
        operation_type : str
            Type of operation ('reception', 'delivery', 'purchase_planning')
        **kwargs
            Additional parameters for specific operations
            
        Returns
        -------
        Dict[str, Any]
            Results of vectorized operations
        """
        results = {}
        
        if operation_type == "reception":
            # Batch process country product reception
            for country_id, country in countries.items():
                country_results = self._process_country_reception(country, **kwargs)
                results[country_id] = country_results
        
        elif operation_type == "purchase_planning":
            # Batch process country purchase planning
            for country_id, country in countries.items():
                country_results = self._process_country_purchase_planning(country, **kwargs)
                results[country_id] = country_results
        
        return results
    
    def _process_country_reception(self, country, sc_network, transport_network, 
                                  sectors_no_transport_network, **kwargs):
        """Process product reception for a single country."""
        country.reset_indicators()
        deliveries = {}
        
        for supplier, _ in sc_network.in_edges(country):
            commercial_link = sc_network[supplier][country]['object']
            
            if commercial_link.use_transport_network:
                country.receive_shipment_and_pay(commercial_link, transport_network)
            else:
                country.receive_service_and_pay(commercial_link)
            
            if commercial_link.delivery > 0:
                deliveries[commercial_link.product] = commercial_link.delivery
        
        return {'deliveries': deliveries}
    
    def _process_country_purchase_planning(self, country, **kwargs):
        """Process purchase planning for a single country."""
        # Countries typically have simpler purchase planning
        # This can be extended with vectorized logic if needed
        return {'purchase_plan': getattr(country, 'purchase_plan', {})}


class VectorizedAgentUpdater:
    """
    High-level orchestrator for vectorized agent operations.
    
    This class coordinates vectorized updates across different agent types
    and provides the main interface for replacing loop-based operations.
    """
    
    def __init__(self):
        self.inventory_manager = VectorizedInventoryManager()
        self.production_manager = VectorizedProductionManager()
        self.household_manager = VectorizedHouseholdManager()
        self.country_manager = VectorizedCountryManager()
    
    @profile_function("Vectorized Agent Reception")
    def vectorized_receive_products(self, agents: Dict, sc_network, transport_network, 
                                   sectors_no_transport_network: List[str]) -> Dict[str, Any]:
        """
        Vectorized implementation of the product reception phase.
        
        This replaces the sequential loop in BaseAgents.receive_products()
        with batch operations on agent collections and cached network topology.
        
        Parameters
        ----------
        agents : Dict
            Dictionary of agent_id -> agent objects
        sc_network : ScNetwork
            Supply chain network
        transport_network : TransportNetwork
            Transport network
        sectors_no_transport_network : List[str]
            Sectors that don't use transport
            
        Returns
        -------
        Dict[str, Any]
            Results of vectorized operations including updated inventories,
            stocks, and performance metrics
        """
        results = {
            'updated_inventories': {},
            'updated_stocks': {},
            'deliveries': {},
            'payments': {},
            'performance_metrics': {}
        }
        
        # Get topology cache for optimized network traversal
        topology_cache = get_topology_cache()
        
        # Use cached topology if available, fallback to network traversal
        use_cache = topology_cache.cache_built
        
        # Collect all deliveries in batch
        all_deliveries = defaultdict(dict)
        all_payments = {}
        
        for agent_id, agent in agents.items():
            # Reset agent indicators (vectorizable)
            if hasattr(agent, 'reset_indicators'):
                agent.reset_indicators()
            
            # Get suppliers using cached topology or network traversal
            if use_cache:
                suppliers_data = topology_cache.get_suppliers(agent_id)
                supplier_links = []
                
                for supplier, edge_data in suppliers_data:
                    commercial_link = topology_cache.get_commercial_link(supplier.pid, agent_id)
                    if commercial_link:
                        supplier_links.append((supplier, commercial_link))
            else:
                # Fallback to direct network traversal
                supplier_links = []
                for supplier, _ in sc_network.in_edges(agent):
                    commercial_link = sc_network[supplier][agent]['object']
                    supplier_links.append((supplier, commercial_link))
            
            # Process commercial links - exactly mirror the sequential logic
            for supplier, commercial_link in supplier_links:
                # Process delivery based on transport requirements (exactly like sequential)
                if commercial_link.use_transport_network:
                    # Transport-based delivery - use original method to ensure proper shipment removal
                    if hasattr(agent, 'receive_shipment_and_pay'):
                        agent.receive_shipment_and_pay(commercial_link, transport_network)
                else:
                    # Service delivery - use original method to ensure proper processing
                    if hasattr(agent, 'receive_service_and_pay'):
                        agent.receive_service_and_pay(commercial_link)
                
                # Collect delivery data for batch processing (use original delivery amount)
                if commercial_link.delivery > 0:
                    all_deliveries[agent_id][commercial_link.product] = commercial_link.delivery
                    all_payments[f"{agent_id}_{supplier.pid}"] = commercial_link.payment
        
        # Batch update inventories
        if all_deliveries:
            results['updated_inventories'] = self.inventory_manager.batch_update_inventories(
                agents, dict(all_deliveries)
            )
        
        # Apply inventory updates back to agents
        for agent_id, new_inventory in results['updated_inventories'].items():
            if agent_id in agents:
                agent = agents[agent_id]
                if hasattr(agent, 'inventory_manager'):
                    agent.inventory_manager.inventory = new_inventory
                elif hasattr(agent, 'inventory'):
                    agent.inventory = new_inventory
        
        results['deliveries'] = dict(all_deliveries)
        results['payments'] = all_payments
        results['performance_metrics']['cache_used'] = use_cache
        
        return results
    
    @profile_function("Vectorized Purchase Planning")
    def vectorized_purchase_planning(self, agents: Dict, adaptive_inventories: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Vectorized implementation of purchase planning across agents.
        
        Parameters
        ----------
        agents : Dict
            Dictionary of agent_id -> agent objects
        adaptive_inventories : bool
            Whether to use adaptive inventory targets
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Purchase plans: agent_id -> {supplier_id -> quantity}
        """
        return self.inventory_manager.batch_calculate_purchase_plans(agents, adaptive_inventories)
    
    @profile_function("Vectorized Production Planning")
    def vectorized_production_planning(self, firms: Dict) -> Dict[str, float]:
        """
        Vectorized implementation of production planning for firms.
        
        Parameters
        ----------
        firms : Dict
            Dictionary of firm_id -> firm objects
            
        Returns
        -------
        Dict[str, float]
            Production targets: firm_id -> production_target
        """
        return self.production_manager.batch_production_planning(firms)


# Global instance for use across the codebase
vectorized_updater = VectorizedAgentUpdater()


def evaluate_inventory_duration_vectorized(input_needs: np.ndarray, stocks: np.ndarray) -> np.ndarray:
    """
    Vectorized version of inventory duration calculation.
    
    Parameters
    ----------
    input_needs : np.ndarray
        Array of input needs per time period
    stocks : np.ndarray
        Array of current stock levels
        
    Returns
    -------
    np.ndarray
        Array of inventory durations (days of supply)
    """
    epsilon = 1e-6
    
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        durations = np.where(
            input_needs > epsilon,
            stocks / input_needs,
            np.where(stocks > 0, np.inf, 0)
        )
    
    return durations


def purchase_planning_function_vectorized(needs: np.ndarray, inventories: np.ndarray, 
                                        targets: np.ndarray, restoration_times: np.ndarray) -> np.ndarray:
    """
    Vectorized version of purchase planning calculation.
    
    Parameters
    ----------
    needs : np.ndarray
        Array of input needs per time period
    inventories : np.ndarray
        Array of current inventory levels
    targets : np.ndarray
        Array of target inventory durations
    restoration_times : np.ndarray
        Array of inventory restoration times
        
    Returns
    -------
    np.ndarray
        Array of purchase quantities needed
    """
    target_stocks = needs * targets
    stock_deficits = np.maximum(0, target_stocks - inventories)
    purchase_quantities = stock_deficits / restoration_times
    
    return purchase_quantities