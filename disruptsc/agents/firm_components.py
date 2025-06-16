import logging
from typing import TYPE_CHECKING, Dict, Any

import numpy as np

from disruptsc.model.basic_functions import rescale_monetary_values

if TYPE_CHECKING:
    from disruptsc.network.sc_network import ScNetwork
    from disruptsc.network.commercial_link import CommercialLink

EPSILON = 1e-6


class ProductionManager:
    """
    Manages production-related operations for a firm.
    
    Handles production planning, capacity management, and production execution.
    """
    
    def __init__(self, firm):
        self.firm = firm
        self.eq_production = 0.0
        self.production = 0.0
        self.production_target = 0.0
        self.eq_production_capacity = 0.0
        self.production_capacity = 0.0
        self.current_production_capacity = 0.0
        self.utilization_rate = 0.8
        self.product_stock = 0.0
        
        # Disruption-related
        self.production_capacity_reduction = 0.0
        self.remaining_disrupted_time = 0.0

    def initialize(self, eq_production: float, utilization_rate: float):
        """Initialize production parameters."""
        self.production_target = eq_production
        self.production = self.production_target
        self.eq_production = self.production_target
        self.utilization_rate = utilization_rate
        self.eq_production_capacity = self.production_target / self.utilization_rate
        self.production_capacity = self.eq_production_capacity

    def decide_production_plan(self, total_order: float):
        """Decide how much to produce based on orders and current stock."""
        self.production_target = max(0.0, total_order - self.product_stock)

    def evaluate_capacity(self, capital_destroyed: float, capital_initial: float):
        """Evaluate current production capacity considering capital destruction."""
        if capital_destroyed > EPSILON:
            self.production_capacity_reduction = capital_destroyed / capital_initial
            logging.debug(f"{self.firm.id_str()} - due to capital destruction, "
                          f"my production capacity is reduced by {self.production_capacity_reduction}")
        else:
            self.production_capacity_reduction = 0
        self.current_production_capacity = self.production_capacity * (1 - self.production_capacity_reduction)

    def get_spare_production_potential(self, inventory: Dict, input_mix: Dict, total_order: float):
        """Calculate spare production capacity."""
        if len(input_mix) == 0:  # If no need for inputs
            potential_production = self.current_production_capacity
        else:
            max_production = production_function(inventory, input_mix)
            potential_production = min([max_production, self.current_production_capacity])
        return max(0, potential_production - (total_order - self.product_stock))

    def produce(self, inventory: Dict, input_mix: Dict, mode="Leontief"):
        """Execute production and update inventory."""
        # Produce
        if len(input_mix) == 0:  # If no need for inputs
            self.production = min([self.production_target, self.current_production_capacity])
        else:
            max_production = production_function(inventory, input_mix, mode)
            self.production = min([max_production, self.production_target, self.current_production_capacity])

        # Add to stock of finished goods
        self.product_stock += self.production

        # Remove input used from inventories (return updated inventory)
        if mode == "Leontief":
            input_used = {input_id: self.production * mix for input_id, mix in input_mix.items()}
            updated_inventory = {input_id: quantity - input_used[input_id]
                                for input_id, quantity in inventory.items()}
            return updated_inventory
        else:
            raise ValueError("Wrong mode chosen")

    def disrupt_production_capacity(self, disruption_duration: int, reduction: float):
        """Apply production capacity disruption."""
        self.remaining_disrupted_time = disruption_duration
        self.production_capacity_reduction = reduction
        logging.info(f'The production capacity of firm {self.firm.pid} is reduced by {reduction} '
                     f'for {disruption_duration} time steps')

    def update_disrupted_production_capacity(self):
        """Update disruption status."""
        is_back_to_normal = self.remaining_disrupted_time == 1
        if self.remaining_disrupted_time > 0:
            self.remaining_disrupted_time -= 1
        if is_back_to_normal:
            self.production_capacity_reduction = 0
            logging.info(f'The production capacity of firm {self.firm.pid} is back to normal')

    def reset(self):
        """Reset production variables."""
        self.production = 0.0
        self.production_target = 0.0
        self.production_capacity = self.eq_production_capacity
        self.product_stock = 0.0


class InventoryManager:
    """
    Manages inventory-related operations for a firm.
    
    Handles inventory tracking, purchase planning, and inventory targets.
    """
    
    def __init__(self, firm):
        self.firm = firm
        self.inventory = {}
        self.inventory_duration_target = {}
        self.inventory_restoration_time = 1
        self.current_inventory_duration = {}
        self.input_needs = {}
        self.eq_needs = {}
        self.purchase_plan = {}
        self.purchase_plan_per_input = {}

    def initialize(self, input_needs: Dict, min_inventory_duration_target: float):
        """Initialize inventory parameters."""
        self.input_needs = input_needs
        self.eq_needs = input_needs.copy()
        self.inventory_duration_target = {
            input_id: min_inventory_duration_target for input_id in input_needs.keys()
        }
        self.inventory = {
            input_id: need * min_inventory_duration_target
            for input_id, need in input_needs.items()
        }

    def evaluate_input_needs(self, input_mix: Dict, production_target: float):
        """Calculate input needs based on production target."""
        self.input_needs = {
            input_pid: input_mix[input_pid] * production_target
            for input_pid, mix in input_mix.items()
        }

    def decide_purchase_plan(self, adaptive_inventories: bool, adapt_weight_based_on_satisfaction: bool,
                           suppliers: Dict):
        """Decide purchase plan for each input and supplier."""
        if adaptive_inventories:
            ref_input_needs = self.input_needs
        else:
            ref_input_needs = self.eq_needs

        # Evaluate current safety days
        self.current_inventory_duration = {
            input_id: (evaluate_inventory_duration(ref_input_needs[input_id], stock)
                       if input_id in ref_input_needs.keys() else 0)
            for input_id, stock in self.inventory.items()
        }

        # Alert if there is less than a day of an input
        for input_id, inventory_duration in self.current_inventory_duration.items():
            if inventory_duration is not None:
                if inventory_duration < 1 - EPSILON:
                    logging.debug(f"{self.firm.id_str()} - Less than 1 time step of inventory for input type {input_id}: "
                                  f"{inventory_duration} vs. {self.inventory_duration_target[input_id]}")

        # Evaluate purchase plan for each sector
        self.purchase_plan_per_input = {
            input_id: purchase_planning_function(need, self.inventory[input_id],
                                                 self.inventory_duration_target[input_id],
                                                 self.inventory_restoration_time)
            for input_id, need in ref_input_needs.items()
        }

        # Handle weight adaptation based on satisfaction
        if adapt_weight_based_on_satisfaction:
            self._adapt_weights_based_on_satisfaction(suppliers)

        # Calculate purchase plan for each supplier
        self.purchase_plan = {
            supplier_id: self.purchase_plan_per_input[info['sector']] * info['weight']
            for supplier_id, info in suppliers.items()
        }

    def _adapt_weights_based_on_satisfaction(self, suppliers: Dict):
        """Adapt supplier weights based on satisfaction levels."""
        from disruptsc.model.basic_functions import generate_weights_from_list
        
        for sector, need in self.purchase_plan_per_input.items():
            suppliers_from_this_sector = [pid for pid, supplier_info in suppliers.items()
                                          if supplier_info['sector'] == sector]
            change_in_satisfaction = False
            for pid, supplier_info in suppliers.items():
                if supplier_info['sector'] == sector:
                    if supplier_info['satisfaction'] < 1 - EPSILON:
                        change_in_satisfaction = True
                        break

            if change_in_satisfaction:
                modified_weights = generate_weights_from_list([
                    supplier_info['satisfaction'] * supplier_info['weight']
                    for pid, supplier_info in suppliers.items()
                    if pid in suppliers_from_this_sector
                ])
                for i, modified_weight in enumerate(modified_weights):
                    suppliers[suppliers_from_this_sector[i]]['weight'] = modified_weight

    def add_to_inventory(self, product_id: str, quantity: float):
        """Add quantity to inventory."""
        if product_id in self.inventory:
            self.inventory[product_id] += quantity
        else:
            self.inventory[product_id] = quantity

    def reset(self):
        """Reset inventory variables."""
        self.inventory = {}
        self.input_needs = {}
        self.current_inventory_duration = {}
        self.purchase_plan = {}


class FinanceManager:
    """
    Manages financial operations for a firm.
    
    Handles revenue, costs, profits, and pricing decisions.
    """
    
    def __init__(self, firm):
        self.firm = firm
        self.eq_finance = {"sales": 0.0, 'costs': {"input": 0.0, "transport": 0.0, "other": 0.0}}
        self.finance = {"sales": 0.0, 'costs': {"input": 0.0, "transport": 0.0, "other": 0.0}}
        self.eq_profit = 0.0
        self.profit = 0.0
        self.eq_price = 1.0
        self.delta_price_input = 0.0
        self.target_margin = 0.2
        self.transport_share = 0.2
        self.capital_initial = 0.0
        self.capital_to_value_added_ratio = 4
        self.capital_destroyed = 0.0

    def initialize_financial_variables(self, eq_production, eq_input_cost, eq_transport_cost, eq_other_cost):
        """Initialize financial parameters."""
        self.eq_finance['sales'] = eq_production
        self.eq_finance['costs']['input'] = eq_input_cost
        self.eq_finance['costs']['transport'] = eq_transport_cost
        self.eq_finance['costs']['other'] = eq_other_cost
        self.eq_profit = self.eq_finance['sales'] - sum(self.eq_finance['costs'].values())
        self.finance = self.eq_finance.copy()
        self.profit = self.eq_profit
        self.delta_price_input = 0

    def initialize_capital(self, eq_production: float, time_resolution: str):
        """Initialize capital based on production value."""
        yearly_eq_production = rescale_monetary_values(eq_production, input_time_resolution=time_resolution,
                                                       target_time_resolution="year")
        self.capital_initial = self.capital_to_value_added_ratio * yearly_eq_production

    def calculate_price(self, sc_network: "ScNetwork"):
        """Calculate price changes due to input cost changes."""
        if self._check_if_supplier_changed_price(sc_network):
            eq_unitary_input_cost, est_unitary_input_cost_at_current_price = self._get_input_costs(sc_network)
            self.delta_price_input = est_unitary_input_cost_at_current_price - eq_unitary_input_cost
            if np.isnan(self.delta_price_input):
                logging.error(f"delta_price_input is NaN: {est_unitary_input_cost_at_current_price} - {eq_unitary_input_cost}")
            logging.debug(f'Firm {self.firm.pid}: Input prices have changed, I set my price to '
                          f'{self.eq_price * (1 + self.delta_price_input):.4f} instead of {self.eq_price}')
        else:
            self.delta_price_input = 0

    def _get_input_costs(self, sc_network: "ScNetwork"):
        """Calculate theoretical input costs."""
        eq_unitary_input_cost = 0
        est_unitary_input_cost_at_current_price = 0
        for edge in sc_network.in_edges(self.firm):
            commercial_link = sc_network[edge[0]][self.firm]['object']
            weight = sc_network[edge[0]][self.firm]['weight']
            eq_unitary_input_cost += commercial_link.eq_price * weight
            est_unitary_input_cost_at_current_price += commercial_link.price * weight
        return eq_unitary_input_cost, est_unitary_input_cost_at_current_price

    def _check_if_supplier_changed_price(self, sc_network: "ScNetwork"):
        """Check if any supplier changed their price."""
        for edge in sc_network.in_edges(self.firm):
            commercial_link = sc_network[edge[0]][self.firm]['object']
            if abs(commercial_link.price - commercial_link.eq_price) > 1e-6:
                return True
        return False

    def evaluate_profit(self, sc_network: "ScNetwork"):
        """Evaluate profit based on sales and costs."""
        # Collect all payments received
        self.finance['sales'] = sum([
            sc_network[self.firm][edge[1]]['object'].payment
            for edge in sc_network.out_edges(self.firm)
        ])
        # Collect all payments made
        self.finance['costs']['input'] = sum([
            sc_network[edge[0]][self.firm]['object'].payment
            for edge in sc_network.in_edges(self.firm)
        ])
        # Compute profit
        self.profit = (self.finance['sales']
                       - self.finance['costs']['input']
                       - self.finance['costs']['other']
                       - self.finance['costs']['transport'])

        self._log_margin_discrepancies()

    def _log_margin_discrepancies(self):
        """Log any discrepancies in margins."""
        expected_gross_margin_no_transport = 1 - sum(list(self.firm.input_mix.values()))
        if self.finance['sales'] > EPSILON:
            realized_gross_margin_no_transport = ((self.finance['sales'] - self.finance['costs']['input'])
                                                  / self.finance['sales'])
            realized_margin = self.profit / self.finance['sales']
        else:
            realized_gross_margin_no_transport = 0
            realized_margin = 0

        # Log discrepancies
        if abs(realized_gross_margin_no_transport - expected_gross_margin_no_transport) > 1e-6:
            logging.debug('Firm ' + str(self.firm.pid) + ': realized gross margin without transport is ' +
                          '{:.3f}'.format(realized_gross_margin_no_transport) + " instead of " +
                          '{:.3f}'.format(expected_gross_margin_no_transport))

        if abs(realized_margin - self.target_margin) > 1e-6:
            logging.debug('Firm ' + str(self.firm.pid) + ': my margin differs from the target one: ' +
                          '{:.3f}'.format(realized_margin) + ' instead of ' + str(self.target_margin))

    def incur_capital_destruction(self, amount: float):
        """Apply capital destruction."""
        if amount > self.capital_initial:
            logging.warning(f"{self.firm.id_str()} - initial capital is lower than destroyed capital "
                            f"({self.capital_initial} vs. {amount})")
            self.capital_destroyed = self.capital_initial
        else:
            self.capital_destroyed = amount

    def record_transport_cost(self, client_id, relative_transport_cost_change, clients: Dict):
        """Record transport cost changes."""
        self.finance['costs']['transport'] += \
            self.eq_finance['costs']['transport'] \
            * clients[client_id]['transport_share'] \
            * (1 + relative_transport_cost_change)

    def calculate_relative_price_change_transport(self, relative_transport_cost_change):
        """Calculate price change due to transport cost changes."""
        return self.eq_finance['costs']['transport'] \
            * relative_transport_cost_change \
            / ((1 - self.target_margin) * self.eq_finance['sales'])

    def reset(self):
        """Reset financial variables."""
        self.finance = {"sales": 0.0, 'costs': {"input": 0.0, "transport": 0.0, "other": 0.0}}
        self.profit = 0.0
        self.delta_price_input = 0.0
        self.capital_destroyed = 0.0


class SupplierManager:
    """
    Manages supplier relationships for a firm.
    
    Handles supplier selection, orders, and relationship tracking.
    """
    
    def __init__(self, firm):
        self.firm = firm
        self.suppliers = {}
        self.clients = {}
        self.order_book = {}
        self.total_order = 0.0
        self.eq_total_order = 0.0
        self.reconstruction_demand = 0.0
        self.reconstruction_produced = 0.0

    def aggregate_orders(self, log_info=False):
        """Aggregate orders from all clients."""
        self.total_order = sum([order for client_pid, order in self.order_book.items()])
        if log_info and self.total_order == 0:
            logging.debug(f'Firm {self.firm.pid} ({self.firm.region_sector}): noone ordered to me')

    def calculate_client_share_in_sales(self):
        """Calculate each client's share in total sales."""
        self.total_order = sum([order for client_pid, order in self.order_book.items()])
        total_qty_km = sum([
            info['distance'] * self.order_book[client_pid]
            for client_pid, info in self.clients.items()
        ])

        # If no one ordered, share is 0
        if self.total_order == 0:
            for client_pid, info in self.clients.items():
                info['share'] = 0
                info['transport_share'] = 0
        # If distance is 0, equal share of transport
        elif total_qty_km == 0:
            nb_active_clients = sum([order > 0 for client_pid, order in self.order_book.items()
                                     if client_pid != "reconstruction"])
            for client_pid, info in self.clients.items():
                info['share'] = self.order_book[client_pid] / self.total_order
                info['transport_share'] = 1 / nb_active_clients
        # Standard case
        else:
            for client_pid, info in self.clients.items():
                info['share'] = self.order_book[client_pid] / self.total_order
                info['transport_share'] = (self.order_book[client_pid] * self.clients[client_pid]['distance']
                                          / total_qty_km)

    def add_reconstruction_order_to_order_book(self):
        """Add reconstruction demand to order book."""
        self.order_book["reconstruction"] = self.reconstruction_demand

    def reset(self):
        """Reset supplier variables."""
        self.order_book = {}
        self.total_order = 0.0


# Utility functions

def production_function(inputs, input_mix, function_type="Leontief"):
    """Production function determining maximum production given inputs."""
    if function_type == "Leontief":
        try:
            return min([inputs[input_id] / input_mix[input_id] for input_id, val in input_mix.items()])
        except KeyError:
            return 0
    else:
        raise ValueError("Wrong mode selected")


def purchase_planning_function(estimated_need: float, inventory: float, inventory_duration_target: float,
                               inventory_restoration_time: float):
    """Decide the quantity of each input to buy according to a dynamical rule."""
    target_inventory = inventory_duration_target * estimated_need
    return max(0.0, estimated_need + 1 / inventory_restoration_time * (target_inventory - inventory))


def evaluate_inventory_duration(estimated_need, inventory):
    """Evaluate how many days of inventory are available."""
    if estimated_need == 0:
        return None
    else:
        return inventory / estimated_need