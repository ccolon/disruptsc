import logging
import random

import numpy as np
from shapely.geometry import Point

from class_agent import Agent
from class_commerciallink import CommercialLink
from functions import purchase_planning_function, production_function, \
    evaluate_inventory_duration, generate_weights, \
    compute_distance_from_arcmin, rescale_values, \
    agent_receive_products_and_pay


# TODO: create class FirmList, CountryList, HouseholdList from userlist as DisruptionList
class Firm(Agent):

    def __init__(self, pid, odpoint=0, sector=0, sector_type=None, input_mix=None, target_margin=0.2,
                 utilization_rate=0.8,
                 importance=1, long=None, lat=None, geometry=None,
                 suppliers=None, clients=None, production=0, inventory_duration_target=1, reactivity_rate=1,
                 usd_per_ton=2864):
        super().__init__(
            agent_type="firm",
            pid=pid,
            odpoint=odpoint,
            long=long,
            lat=lat
        )
        # Parameters depending on data
        self.usd_per_ton = usd_per_ton
        self.geometry = geometry
        self.importance = importance
        self.sector = sector
        self.sector_type = sector_type
        self.input_mix = input_mix or {}

        # Free parameters
        if input_mix is None:
            self.inventory_duration_target = inventory_duration_target
        else:
            self.inventory_duration_target = {key: inventory_duration_target for key in input_mix.keys()}
        self.reactivity_rate = reactivity_rate
        self.eq_production_capacity = production / utilization_rate
        self.utilization_rate = utilization_rate
        self.target_margin = target_margin

        # Parameters depending on supplier-buyer network
        self.suppliers = suppliers or {}
        self.clients = clients or {}

        # Parameters sets at initialization
        self.eq_finance = {"sales": 0, 'costs': {"input": 0, "transport": 0, "other": 0}}
        self.eq_profit = 0
        self.eq_price = 1
        self.eq_total_order = 0

        # Variables, all initialized
        self.production = production
        self.production_target = production
        self.production_capacity = production / utilization_rate
        self.purchase_plan = {}
        self.order_book = {}
        self.total_order = 0
        self.input_needs = {}
        self.rationing = 1
        self.eq_needs = {}
        self.current_inventory_duration = {}
        self.inventory = {}
        self.product_stock = 0
        self.profit = 0
        self.finance = {"sales": 0, 'costs': {"input": 0, "transport": 0, "other": 0}}
        self.delta_price_input = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

        # Disruption
        self.remaining_disrupted_time = 0
        self.production_capacity_reduction = 0

    def reset_variables(self):
        self.eq_finance = {"sales": 0, 'costs': {"input": 0, "transport": 0, "other": 0}}
        self.eq_profit = 0
        self.eq_price = 1
        self.production = 0
        self.production_target = 0
        self.production_capacity = self.eq_production_capacity
        self.purchase_plan = {}
        self.order_book = {}
        self.total_order = 0
        self.input_needs = {}
        self.rationing = 1
        self.eq_needs = {}
        self.current_inventory_duration = {}
        self.inventory = {}
        self.product_stock = 0
        self.profit = 0
        self.finance = {"sales": 0, 'costs': {"input": 0, "transport": 0, "other": 0}}
        self.delta_price_input = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

    def update_production_capacity(self):
        is_back_to_normal = self.remaining_disrupted_time == 1  # Identify those who will be back to normal
        if self.remaining_disrupted_time > 0:  # Update the remaining time in disruption
            self.remaining_disrupted_time -= 1
        if is_back_to_normal:  # Update the remaining time in disruption
            self.production_capacity_reduction = 0
            logging.info(f'The production capacity of firm {self.pid} is back to normal')

    def reduce_production_capacity(self, disruption_duration: int, reduction: float):
        self.remaining_disrupted_time = disruption_duration
        self.production_capacity_reduction = reduction
        logging.info(f'The production capacity of firm {self.pid} is reduced by {reduction} '
                     f'for {disruption_duration} time steps')


    def initialize_ope_var_using_eq_production(self, eq_production):
        self.production_target = eq_production
        self.production = self.production_target
        self.eq_production_capacity = self.production_target / self.utilization_rate
        self.production_capacity = self.eq_production_capacity
        self.evaluate_input_needs()
        self.eq_needs = self.input_needs
        self.inventory = {
            input_id: need * (1 + self.inventory_duration_target[input_id])
            for input_id, need in self.input_needs.items()
        }
        self.decide_purchase_plan()

    def initialize_fin_var_using_eq_cost(self, eq_production, eq_input_cost,
                                         eq_transport_cost, eq_other_cost):
        self.eq_finance['sales'] = eq_production
        self.eq_finance['costs']['input'] = eq_input_cost
        self.eq_finance['costs']['transport'] = eq_transport_cost
        self.eq_finance['costs']['other'] = eq_other_cost
        self.eq_profit = self.eq_finance['sales'] - sum(self.eq_finance['costs'].values())
        self.finance['sales'] = self.eq_finance['sales']
        self.finance['costs']['input'] = self.eq_finance['costs']['input']
        self.finance['costs']['transport'] = self.eq_finance['costs']['transport']
        self.finance['costs']['other'] = self.eq_finance['costs']['other']
        self.profit = self.eq_profit
        self.delta_price_input = 0

    def add_noise_to_geometry(self, noise_level=1e-5):
        self.geometry = Point(self.long + noise_level * random.uniform(0, 1),
                              self.lat + noise_level * random.uniform(0, 1))

    def distance_to_other(self, other_firm):
        if (self.odpoint == -1) or (other_firm.odpoint == -1):  # if virtual firms
            return 1
        else:
            return compute_distance_from_arcmin(self.long, self.lat, other_firm.long, other_firm.lat)

    def select_suppliers_from_data(self, graph, firm_list: list, country_list,
                                   inputed_supplier_links, output, import_code='IMP'):

        for inputed_supplier_link in list(inputed_supplier_links.transpose().to_dict().values()):
            # Create an edge in the graph
            supplier_id = inputed_supplier_link['supplier_id']
            product_sector = inputed_supplier_link['product_sector']
            supplier_object = firm_list[supplier_id]
            graph.add_edge(supplier_object, self,
                           object=CommercialLink(
                               pid=str(supplier_id) + "->" + str(self.pid),
                               product=product_sector,
                               product_type=supplier_object.sector_type,
                               essential=inputed_supplier_link['is_essential'],
                               category='domestic_B2B',
                               supplier_id=supplier_id,
                               buyer_id=self.pid)
                           )
            # Associate a weight to the edge
            weight_in_input_mix = inputed_supplier_link['transaction'] / output
            graph[supplier_object][self]['weight'] = weight_in_input_mix
            # The firm saves the name of the supplier, its sector, its weight among firm of the same sector (without I/O technical coefficient)
            total_input_same_sector = inputed_supplier_links.loc[
                inputed_supplier_links['product_sector'] == product_sector, "transaction"].sum()
            weight_among_same_product = inputed_supplier_link['transaction'] / total_input_same_sector
            self.suppliers[supplier_id] = {'sector': product_sector, 'weight': weight_among_same_product}

    def select_suppliers(self, graph, firm_list: list, country_list,
                         nb_suppliers_per_input=1, weight_localization=1, import_code='IMP'):
        """
        The firm selects its suppliers.

        The firm checks its input mix to identify which type of inputs are needed.
        For each type of input, it selects the appropriate number of suppliers.
        Choice of suppliers is random, based on distance to eligible suppliers and 
        their importance.

        If imports are needed, the firms select a country as supplier. Choice is 
        random, based on the country's importance.

        Parameters
        ----------
        graph : networkx.DiGraph
            Supply chain graph
        firm_list : list of Firms
            Generated by createFirms function
        country_list : list of Countries
            Generated by createCountriesfunction
        nb_suppliers_per_input : float between 1 and 2
            Nb of suppliers per type of inputs. If it is a decimal between 1 and 2,
            some firms will have 1 supplier, other 2 suppliers, such that the
            average matches the specified value.
        weight_localization : float
            Give weight to distance when choosing supplier. The larger, the closer
            the suppliers will be selected.
        import_code : string
            Code that identify imports in the input mix.

        Returns
        -------
        int
            0

        """
        for sector_id, sector_weight in self.input_mix.items():

            # If it is imports, identify international suppliers and calculate
            # their probability to be chosen, which is based on importance.
            if sector_id == import_code:
                # Identify countries as suppliers if the corresponding sector does export
                importance_threshold = 1e-6
                potential_supplier_pid = [
                    country.pid
                    for country in country_list
                    if country.supply_importance > importance_threshold
                ]
                importance_of_each = [
                    country.supply_importance
                    for country in country_list
                    if country.supply_importance > importance_threshold
                ]
                prob_to_be_selected = np.array(importance_of_each)
                prob_to_be_selected /= prob_to_be_selected.sum()

            # For the other types of inputs, identify the domestic suppliers, and
            # calculate their probability to be chosen, based on distance and importance
            else:
                potential_supplier_pid = [firm.pid for firm in firm_list if
                                          firm.sector == sector_id]  # Identify the id of potential suppliers among
                # the other firms
                if sector_id == self.sector:
                    potential_supplier_pid.remove(self.pid)  # remove oneself
                if len(potential_supplier_pid) == 0:
                    raise ValueError("Firm " + str(self.pid) +
                                     ": no potential supplier for input " + str(sector_id))
                # print("\n", self.pid, ":", str(len(potential_supplier_pid)), "for sector", sector_id)
                # print([
                #     self.distance_to_other(firm_list[firm_pid]) 
                #     for firm_pid in potential_supplier_pid
                # ])
                distance_to_each = rescale_values([
                    self.distance_to_other(firm_list[firm_pid])
                    for firm_pid in potential_supplier_pid
                ])  # Compute distance to each of them (vol d oiseau)
                # print(distance_to_each)
                importance_of_each = rescale_values([firm_list[firm_pid].importance for firm_pid in
                                                     potential_supplier_pid])  # Get importance for each of them
                prob_to_be_selected = np.array(importance_of_each) / (np.array(distance_to_each) ** weight_localization)
                prob_to_be_selected /= prob_to_be_selected.sum()

            # Determine the numbre of supplier(s) to select. 1 or 2.
            if random.uniform(0, 1) < nb_suppliers_per_input - 1:
                nb_suppliers_to_choose = 2
                if nb_suppliers_to_choose > len(potential_supplier_pid):
                    nb_suppliers_to_choose = 1
            else:
                nb_suppliers_to_choose = 1

            # Select the supplier(s). It there is 2 suppliers, then we generate 
            # random weight. It determines how much is bought from each supplier.
            selected_supplier_id = np.random.choice(potential_supplier_pid,
                                                    p=prob_to_be_selected, size=nb_suppliers_to_choose,
                                                    replace=False).tolist()
            supplier_weights = generate_weights(
                nb_suppliers_to_choose)  # Generate one random weight per number of supplier, sum to 1

            # For each new supplier, create a new CommercialLink in the supply chain network.
            for supplier_id in selected_supplier_id:
                # Retrieve the appropriate supplier object from the id
                # If it is a country we get it from the country list
                # It it is a firm we get it from the firm list
                if sector_id == import_code:
                    supplier_object = [country for country in country_list if country.pid == supplier_id][0]
                    link_category = 'import'
                    product_type = "imports"
                else:
                    supplier_object = firm_list[supplier_id]
                    link_category = 'domestic_B2B'
                    product_type = firm_list[supplier_id].sector_type
                # Create an edge in the graph
                graph.add_edge(supplier_object, self,
                               object=CommercialLink(
                                   pid=str(supplier_id) + "->" + str(self.pid),
                                   product=sector_id,
                                   product_type=product_type,
                                   category=link_category,
                                   supplier_id=supplier_id,
                                   buyer_id=self.pid)
                               )
                # Associate a weight, which includes the I/O technical coefficient
                supplier_weight = supplier_weights.pop()
                graph[supplier_object][self]['weight'] = sector_weight * supplier_weight
                # The firm saves the name of the supplier, its sector, its weight (without I/O technical coefficient)
                self.suppliers[supplier_id] = {'sector': sector_id, 'weight': supplier_weight}
                # The supplier saves the name of the client, its sector, and distance to it.
                # The share of sales cannot be calculated now
                distance = self.distance_to_other(supplier_object)
                supplier_object.clients[self.pid] = {'sector': self.sector, 'share': 0, 'transport_share': 0,
                                                     'distance': distance}

    def choose_route(self, transport_network,
                     origin_node, destination_node,
                     accepted_logistics_modes, capacity_burden=None):
        """
        The agent choose the delivery route

        If the simple case in which there is only one accepted_logistics_modes
        (as defined by the main parameter logistic_modes)
        then it is simply the shortest_route using the appropriate weigth

        If there are several accepted_logistics_modes, then the agent will investigate different route,
        one per accepted_logistics_mode. They will then pick one, with a certain probability taking into account the
        weight This more complex mode is used when, according to the capacity and cost data, all the exports or
        importzs are using one route, whereas in the data, we observe still some flows using another mode of

        tranpsort. So we use this to "force" some flow to take the other routes.
        """
        # If accepted_logistics_modes is a string, then simply pick the shortest route of this logistic mode
        if isinstance(accepted_logistics_modes, str):
            route = transport_network.provide_shortest_route(origin_node,
                                                             destination_node,
                                                             route_weight=accepted_logistics_modes + "_weight")
            return route, accepted_logistics_modes

        # If it is a list, it means that the agent will chosen between different logistic corridors
        # with a certain probability
        elif isinstance(accepted_logistics_modes, list):
            # pick routes for each modes
            routes = {
                mode: transport_network.provide_shortest_route(origin_node,
                                                               destination_node, route_weight=mode + "_weight")
                for mode in accepted_logistics_modes
            }
            # compute associated weight and capacity_weight
            modes_weight = {
                mode: {
                    mode + "_weight": transport_network.sum_indicator_on_route(route, mode + "_weight"),
                    "weight": transport_network.sum_indicator_on_route(route, "weight"),
                    "capacity_weight": transport_network.sum_indicator_on_route(route, "capacity_weight")
                }
                for mode, route in routes.items()
            }
            # remove any mode which is over capacity (where capacity_weight > capacity_burden)
            for mode, route in routes.items():
                if mode != "intl_rail":
                    if transport_network.check_edge_in_route(route, (2610, 2589)):
                        print("(2610, 2589) in", mode)
            modes_weight = {
                mode: weight_dic['weight']
                for mode, weight_dic in modes_weight.items()
                if weight_dic['capacity_weight'] < capacity_burden
            }
            if len(modes_weight) == 0:
                logging.warning("All transport modes are over capacity, no route selected!")
                return None
            # and select one route choosing random weighted choice
            selection_weights = rescale_values(list(modes_weight.values()), minimum=0, maximum=0.5)
            selection_weights = [1 - w for w in selection_weights]
            selected_mode = random.choices(
                list(modes_weight.keys()),
                weights=selection_weights,
                k=1
            )[0]
            # print("Firm "+str(self.pid)+" chooses "+selected_mode+
            #     " to serve a client located "+str(destination_node))
            route = routes[selected_mode]
            return route, selected_mode

        raise ValueError("The transport_mode attributes of the commerical link\
                          does not belong to ('roads', 'intl_multimodes')")

    def calculate_client_share_in_sales(self):
        # Only works if the order book was computed
        self.total_order = sum([order for client_pid, order in self.order_book.items()])
        total_qty_km = sum([
            info['distance'] * self.order_book[client_pid]
            for client_pid, info in self.clients.items()
        ])
        # self.total_B2B_order = sum([order for client_pid, order in self.order_book.items() if client_pid != -1])
        # If noone ordered to me, share is 0 (avoid division per 0)
        if self.total_order == 0:
            for client_pid, info in self.clients.items():
                info['share'] = 0
                info['transport_share'] = 0

        # If some clients ordered to me, but distance is 0 (no transport), then equal share of transport
        elif total_qty_km == 0:
            nb_active_clients = sum([order > 0 for client_pid, order in self.order_book.items()])
            for client_pid, info in self.clients.items():
                info['share'] = self.order_book[client_pid] / self.total_order
                info['transport_share'] = 1 / nb_active_clients

        # Otherwise, standard case
        else:
            for client_pid, info in self.clients.items():
                info['share'] = self.order_book[client_pid] / self.total_order
                info['transport_share'] = self.order_book[client_pid] * self.clients[client_pid][
                    'distance'] / total_qty_km

    def aggregate_orders(self, print_info=False):
        self.total_order = sum([order for client_pid, order in self.order_book.items()])
        if print_info:
            if self.total_order == 0:
                logging.debug(f'Firm {self.pid} ({self.sector}): no-one ordered to me')

    def decide_production_plan(self):
        self.production_target = self.total_order - self.product_stock

    def calculate_price(self, graph):
        """
        Evaluate the relative increase in price due to changes in input price
        In addition, upon delivery, price will be adjusted for each client to reflect potential rerouting
        """
        if self.check_if_supplier_changed_price(graph):
            # One way to compute it is commented.
            #     self.delta_price_input = self.calculate_input_induced_price_change(graph)
            #     logging.debug('Firm '+str(self.pid)+': Input prices have changed, I set '+
            #     "my price to "+'{:.4f}'.format(self.eq_price*(1+self.delta_price_input))+
            #     " instead of "+str(self.eq_price))

            # I compute how much would be my input cost to produce one unit of output
            # if I had to buy the input at this price
            eq_unitary_input_cost, est_unitary_input_cost_at_current_price = self.get_input_costs(graph)
            # I scale this added cost to my total orders
            self.delta_price_input = est_unitary_input_cost_at_current_price - eq_unitary_input_cost
            if self.delta_price_input is np.nan:
                print(self.delta_price_input)
                print(est_unitary_input_cost_at_current_price)
                print(eq_unitary_input_cost)
            # added_input_cost = (est_unitary_input_cost_at_current_price - eq_unitary_input_cost) * self.total_order
            # self.delta_price_input = added_input_cost / self.total_order
            logging.debug('Firm ' + str(self.pid) + ': Input prices have changed, I set my price to ' +
                          '{:.4f}'.format(self.eq_price * (1 + self.delta_price_input)) +
                          " instead of " + str(self.eq_price))
        else:
            self.delta_price_input = 0

    def get_input_costs(self, graph):
        eq_unitary_input_cost = 0
        est_unitary_input_cost_at_current_price = 0
        for edge in graph.in_edges(self):
            eq_unitary_input_cost += graph[edge[0]][self]['object'].eq_price * graph[edge[0]][self]['weight']
            est_unitary_input_cost_at_current_price += graph[edge[0]][self]['object'].price * graph[edge[0]][self][
                'weight']
        return eq_unitary_input_cost, est_unitary_input_cost_at_current_price

    def evaluate_input_needs(self):
        self.input_needs = {
            input_pid: self.input_mix[input_pid] * self.production_target
            for input_pid, mix in self.input_mix.items()
        }

    def decide_purchase_plan(self, mode="equilibrium"):
        """
        If mode="equilibrium", it aims to come back to equilibrium inventories
        If mode="reactive", it uses current orders to evaluate the target inventories
        """

        if mode == "reactive":
            ref_input_needs = self.input_needs

        elif mode == "equilibrium":
            ref_input_needs = self.eq_needs

        else:
            raise NotImplementedError(f"Mode {mode} not implemented")

        # Evaluate the current safety days
        self.current_inventory_duration = {
            input_id: (evaluate_inventory_duration(ref_input_needs[input_id],
                                                   stock) if input_id in ref_input_needs.keys() else 0)
            for input_id, stock in self.inventory.items()
        }

        # Alert if there is less than a day of an input
        if True:
            for input_id, inventory_duration in self.current_inventory_duration.items():
                if inventory_duration is not None:
                    if inventory_duration < 1 - 1e-6:
                        if -1 in self.clients.keys():
                            sales_to_hh = self.clients[-1]['share'] * self.production_target
                        else:
                            sales_to_hh = 0
                        logging.debug('Firm ' + str(self.pid) + " of sector " + str(
                            self.sector) + " selling to households " + str(
                            sales_to_hh) + " less than 1 day of inventory for input type " + str(input_id))

        # Evaluate purchase plan for each sector
        purchase_plan_per_sector = {
            input_id: purchase_planning_function(need, self.inventory[input_id],
                                                 self.inventory_duration_target[input_id], self.reactivity_rate)
            # input_id: purchase_planning_function(need, self.inventory[input_id],
            # self.inventory_duration_old, self.reactivity_rate)
            for input_id, need in ref_input_needs.items()
        }
        # print(self.input_mix)
        # print(self.eq_needs)
        # print(self.inventory)
        # print(self.inventory_duration_target)
        # print(purchase_plan_per_sector)
        # Deduce the purchase plan for each supplier
        self.purchase_plan = {
            supplier_id: purchase_plan_per_sector[info['sector']] * info['weight']
            for supplier_id, info in self.suppliers.items()
        }

    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            if edge[0].pid in self.purchase_plan.keys():
                quantity_to_buy = self.purchase_plan[edge[0].pid]
                if quantity_to_buy == 0:
                    logging.debug("Firm " + str(self.pid) + ": I am not planning to buy anything from supplier " + str(
                        edge[0].pid))
            else:
                logging.error(
                    "Firm " + str(self.pid) + ": supplier " + str(edge[0].pid) + " is not in my purchase plan")
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

    def retrieve_orders(self, graph):
        for edge in graph.out_edges(self):
            quantity_ordered = graph[self][edge[1]]['object'].order
            quantity_ordered = graph[self][edge[1]]['object'].order
            self.order_book[edge[1].pid] = quantity_ordered

    def produce(self, mode="Leontief"):
        current_production_capacity = self.production_capacity * (1 - self.production_capacity_reduction)
        # Produce
        if len(self.input_mix) == 0:  # If no need for inputs
            self.production = min([self.production_target, current_production_capacity])
        else:
            max_production = production_function(self.inventory, self.input_mix, mode)  # Max prod given inventory
            self.production = min([max_production, self.production_target, current_production_capacity])

        # Add to stock of finished goods
        self.product_stock += self.production

        # Remove input used from inventories
        if mode == "Leontief":
            input_used = {input_id: self.production * mix for input_id, mix in self.input_mix.items()}
            self.inventory = {input_id: quantity - input_used[input_id] for input_id, quantity in
                              self.inventory.items()}
        else:
            raise ValueError("Wrong mode chosen")

    def calculate_input_induced_price_change(self, graph):
        """The firm evaluates the input costs of producting one unit of output if it had to buy the inputs at current
        price It is a theoretical cost, because in simulations it may use inventory
        """
        eq_theoretical_input_cost, current_theoretical_input_cost = self.get_input_costs(graph)
        input_cost_share = eq_theoretical_input_cost / 1
        relative_change = (current_theoretical_input_cost - eq_theoretical_input_cost) / eq_theoretical_input_cost
        return relative_change * input_cost_share / (1 - self.target_margin)

    def check_if_supplier_changed_price(self, graph):  # firms could record the last price they paid their input
        for edge in graph.in_edges(self):
            if abs(graph[edge[0]][self]['object'].price - graph[edge[0]][self]['object'].eq_price) > 1e-6:
                return True
        return False

    def ration_quantity_to_deliver(self):
        # remove rationing as attribute
        pass

    def deliver_products(self, graph, transport_network=None, sectors_no_transport_network=None,
                         rationing_mode="equal",
                         monetary_units_in_model="mUSD",
                         cost_repercussion_mode="type1", explicit_service_firm=True):

        # Do nothing if no orders
        if self.total_order == 0:
            # logging.warning('Firm '+str(self.pid)+' ('+self.sector+'): no one ordered to me')
            return 0

        # Otherwise compute rationing factor
        epsilon = 1e-6
        self.rationing = self.product_stock / self.total_order
        # Check the case in which the firm has too much product to sale
        # It should not happen, hence a warning
        quantity_to_deliver = None
        if self.rationing > 1 + epsilon:
            logging.warning(f'Firm {self.pid}: I have produced too much')
            self.rationing = 1
        # If rationing factor is 1, then it delivers what was ordered
        elif self.rationing >= 1 - epsilon:
            quantity_to_deliver = {buyer_id: order for buyer_id, order in self.order_book.items()}
        # If rationing occurs, then two rationing behavior: equal or household_first
        else:
            logging.debug(f'Firm {self.pid}: I have to ration my clients by {(1 - self.rationing) * 100:.2f}%')
            # If equal, simply apply rationing factor
            if rationing_mode == "equal":
                quantity_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in self.order_book.items()}

            elif rationing_mode == "household_first":
                if -1 not in self.order_book.keys():
                    quantity_to_deliver = {buyer_id: order * self.rationing for buyer_id, order in
                                           self.order_book.items()}
                elif len(self.order_book.keys()) == 1:  # only households order to this firm
                    quantity_to_deliver = {-1: self.total_order}
                else:
                    order_households = self.order_book[-1]
                    if order_households < self.product_stock:
                        remaining_product_stock = self.product_stock - order_households
                        if (self.total_order - order_households) <= 0:
                            logging.warning("Firm " + str(self.pid) + ': ' + str(self.total_order - order_households))
                        rationing_for_business = remaining_product_stock / (self.total_order - order_households)
                        quantity_to_deliver = {buyer_id: order * rationing_for_business for buyer_id, order in
                                               self.order_book.items() if buyer_id != -1}
                        quantity_to_deliver[-1] = order_households
                    else:
                        quantity_to_deliver = {buyer_id: 0 for buyer_id, order in self.order_book.items() if
                                               buyer_id != -1}
                        quantity_to_deliver[-1] = self.product_stock
            else:
                raise ValueError('Wrong rationing_mode chosen')
        # We initialize transport costs, it will be updated for each shipment
        self.finance['costs']['transport'] = 0
        self.generalized_transport_cost = 0
        self.usd_transported = 0
        self.tons_transported = 0
        self.tonkm_transported = 0

        # For each client, we define the quantity to deliver then send the shipment 
        for edge in graph.out_edges(self):
            if graph[self][edge[1]]['object'].order == 0:
                logging.debug("Agent " + str(self.pid) + ": " +
                              str(graph[self][edge[1]]['object'].buyer_id) + " is my client but did not order")
                continue
            graph[self][edge[1]]['object'].delivery = quantity_to_deliver[edge[1].pid]
            graph[self][edge[1]]['object'].delivery_in_tons = \
                Firm.transformUSD_to_tons(quantity_to_deliver[edge[1].pid], monetary_units_in_model, self.usd_per_ton)

            if explicit_service_firm:
                # If the client is B2C (applied only we had one single representative agent for all households)
                if edge[1].pid == -1:
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])
                # If this is service flow, deliver without infrastructure
                elif self.sector_type in sectors_no_transport_network:
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])
                # otherwise use infrastructure
                else:
                    self.send_shipment(
                        graph[self][edge[1]]['object'],
                        transport_network,
                        monetary_units_in_model,
                        cost_repercussion_mode
                    )

            else:
                # If it's B2B and no service client, we send to the transport network, 
                # price will be adjusted according to transport conditions
                if (self.odpoint != -1) and (edge[1].odpoint != -1) and (edge[1].pid != -1):
                    self.send_shipment(
                        graph[self][edge[1]]['object'],
                        transport_network,
                        monetary_units_in_model,
                        cost_repercussion_mode
                    )

                # If it's B2C, or B2B with service client, we send directly, 
                # and adjust price with input costs. There is still transport costs.
                elif (self.odpoint == -1) or (edge[1].odpoint == -1) or (edge[1].pid == -1):
                    self.deliver_without_infrastructure(graph[self][edge[1]]['object'])

                else:
                    logging.error('There should not be this other case.')

    def deliver_without_infrastructure(self, commercial_link):
        """ The firm deliver its products without using transportation infrastructure
        This case applies to service firm and to households 
        Note that we still account for transport cost, proportionnaly to the share of the clients
        Price can be higher than 1, if there are changes in price inputs
        """
        commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
        self.product_stock -= commercial_link.delivery
        self.finance['costs']['transport'] += (self.clients[commercial_link.buyer_id]['share'] *
                                               self.eq_finance['costs']['transport'])

    def send_shipment(self, commercial_link, transport_network,
                      monetary_units_in_model, cost_repercussion_mode):

        monetary_unit_factor = {
            "mUSD": 1e6,
            "kUSD": 1e3,
            "USD": 1
        }
        factor = monetary_unit_factor[monetary_units_in_model]
        # print("send_shipment", 0 in transport_network.nodes)
        """Only apply to B2B flows 
        """
        if len(commercial_link.route) == 0:
            raise ValueError("Firm " + str(self.pid) + " " + str(self.sector) +
                             ": commercial link " + str(commercial_link.pid) + " (qty " + str(
                commercial_link.order) + ")"
                                         " is not associated to any route, I cannot send any shipment to client " +
                             str(commercial_link.buyer_id))

        if self.check_route_availability(commercial_link, transport_network, 'main') == 'available':
            # If the normal route is available, we can send the shipment as usual 
            # and pay the usual price
            commercial_link.price = commercial_link.eq_price * (1 + self.delta_price_input)
            commercial_link.current_route = 'main'
            transport_network.transport_shipment(commercial_link)
            self.product_stock -= commercial_link.delivery
            self.generalized_transport_cost += (commercial_link.route_time_cost
                                                + commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton)
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons * commercial_link.route_length
            self.finance['costs']['transport'] += \
                self.clients[commercial_link.buyer_id]['share'] \
                * self.eq_finance['costs']['transport']
            return 0

        # If there is an alternative route already discovered, 
        # and if this alternative route is available, then we use it
        if (len(commercial_link.alternative_route) > 0) \
                & (self.check_route_availability(commercial_link, transport_network, 'alternative') == 'available'):
            route = commercial_link.alternative_route
        # Otherwise we need to discover a new one
        else:
            route = self.discover_new_route(commercial_link, transport_network)

        # If the alternative route is available, or if we discovered one, we proceed
        if route is not None:
            commercial_link.current_route = 'alternative'
            # Calculate contribution to generalized transport cost, to usd/tons/tonkms transported
            self.generalized_transport_cost += (commercial_link.alternative_route_time_cost
                                                + Firm.transformUSD_to_tons(commercial_link.delivery,
                                                                            monetary_units_in_model, self.usd_per_ton)
                                                * commercial_link.alternative_route_cost_per_ton)
            self.usd_transported += commercial_link.delivery
            self.tons_transported += commercial_link.delivery_in_tons
            self.tonkm_transported += commercial_link.delivery_in_tons * commercial_link.alternative_route_length

            # We translate this real cost into transport cost
            if cost_repercussion_mode == "type1":  # relative cost change with actual bill
                # Calculate relative increase in routing cost
                new_transport_bill = commercial_link.delivery_in_tons * commercial_link.alternative_route_cost_per_ton
                normal_transport_bill = commercial_link.delivery_in_tons * commercial_link.route_cost_per_ton
                # print(self.pid, self.sector_type)
                # print(commercial_link.delivery_in_tons, commercial_link.route_cost_per_ton,
                #       commercial_link.alternative_route_cost_per_ton)
                # print(normal_transport_bill, new_transport_bill)
                relative_cost_change = max(new_transport_bill - normal_transport_bill, 0) / normal_transport_bill
                # If switched transport mode, add switching cost
                switching_cost = 0.5
                if commercial_link.alternative_route_mode != commercial_link.route_mode:
                    relative_cost_change = relative_cost_change + switching_cost
                # Translate that into an increase in transport costs in the balance sheet
                self.finance['costs']['transport'] += \
                    self.eq_finance['costs']['transport'] \
                    * self.clients[commercial_link.buyer_id]['transport_share'] \
                    * (1 + relative_cost_change)
                relative_price_change_transport = \
                    self.eq_finance['costs']['transport'] \
                    * relative_cost_change \
                    / ((1 - self.target_margin) * self.eq_finance['sales'])
                # Calculate the relative price change, including any increase due to the prices of inputs
                total_relative_price_change = self.delta_price_input + relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)

            elif cost_repercussion_mode == "type2":  # actual repercussion de la bill
                added_costUSD_per_ton = max(commercial_link.alternative_route_cost_per_ton -
                                            commercial_link.route_cost_per_ton,
                                            0)
                added_costUSD_per_mUSD = added_costUSD_per_ton / (self.usd_per_ton / factor)
                added_costmUSD_per_mUSD = added_costUSD_per_mUSD / factor
                added_transport_bill = added_costmUSD_per_mUSD * commercial_link.delivery
                self.finance['costs']['transport'] += \
                    self.eq_finance['costs']['transport'] + added_transport_bill
                commercial_link.price = (commercial_link.eq_price
                                         + self.delta_price_input
                                         + added_costmUSD_per_mUSD)
                relative_price_change_transport = \
                    commercial_link.price / (commercial_link.eq_price + self.delta_price_input) - 1
                if (commercial_link.price is None) or (commercial_link.price is np.nan):
                    raise ValueError("Price should be a float, it is " + str(commercial_link.price))

                cost_increase = (commercial_link.alternative_route_cost_per_ton
                                 - commercial_link.route_cost_per_ton) / commercial_link.route_cost_per_ton

                logging.debug(f"Firm {self.pid}"
                              f": qty {commercial_link.delivery_in_tons} tons"
                              f" increase in route cost per ton {cost_increase}"
                              f" increased bill mUSD {added_costmUSD_per_mUSD * commercial_link.delivery}"
                              )

            elif cost_repercussion_mode == "type3":
                relative_cost_change = (commercial_link.alternative_route_time_cost
                                        - commercial_link.route_time_cost) / commercial_link.route_time_cost
                self.finance['costs']['transport'] += (self.eq_finance['costs']['transport']
                                                       * self.clients[commercial_link.buyer_id]['share']
                                                       * (1 + relative_cost_change))
                relative_price_change_transport = (
                        self.eq_finance['costs']['transport']
                        * relative_cost_change
                        / ((1 - self.target_margin) * self.eq_finance['sales']))

                total_relative_price_change = self.delta_price_input + relative_price_change_transport
                commercial_link.price = commercial_link.eq_price * (1 + total_relative_price_change)
            else:
                raise NotImplementedError(f"Type {cost_repercussion_mode} not implemented")

            # If the increase in transport is larger than 2, then we do not deliver the goods
            if relative_price_change_transport > 2:
                logging.info("Firm " + str(self.pid) + ": found an alternative route to " +
                             str(commercial_link.buyer_id) + ", but it is costlier by " +
                             '{:.2f}'.format(100 * relative_price_change_transport) + "%, price would be " +
                             '{:.4f}'.format(commercial_link.price) + " instead of " +
                             '{:.4f}'.format(commercial_link.eq_price * (1 + self.delta_price_input)) +
                             ' so I decide not to send it now.'
                             )
                commercial_link.price = commercial_link.eq_price
                commercial_link.current_route = 'none'
                # commercial_link.delivery = 0
            # Otherwise, we deliver
            else:
                transport_network.transport_shipment(commercial_link)
                self.product_stock -= commercial_link.delivery
                # Print information
                logging.info("Firm " + str(self.pid) + ": found an alternative route to " +
                             str(commercial_link.buyer_id) + ", it is costlier by " +
                             '{:.2f}'.format(100 * relative_price_change_transport) + "%, price is " +
                             '{:.4f}'.format(commercial_link.price) + " instead of " +
                             '{:.4f}'.format(commercial_link.eq_price * (1 + self.delta_price_input))
                             )
        # If we do not find a route, then we do not deliver
        else:
            logging.info('Firm ' + str(self.pid) + ": because of disruption, " +
                         "there is no route between me and firm " + str(commercial_link.buyer_id))
            # We do not write how the input price would have changed
            commercial_link.price = commercial_link.eq_price
            commercial_link.current_route = 'none'
            # We do not pay the transporter, so we don't increment the transport cost
            # We set delivery to 0
            commercial_link.delivery = 0

    def discover_new_route(self, commercial_link, transport_network):
        origin_node = self.odpoint
        destination_node = commercial_link.route[-1][0]
        route, selected_mode = self.choose_route(
            transport_network=transport_network.get_undisrupted_network(),
            origin_node=origin_node,
            destination_node=destination_node,
            accepted_logistics_modes=commercial_link.possible_transport_modes
        )
        # If we find a new route, we save it as the alternative one
        if route is not None:
            commercial_link.storeRouteInformation(
                route=route,
                transport_mode=selected_mode,
                main_or_alternative="alternative",
                transport_network=transport_network
            )
        return route

    def add_congestion_malus2(self, graph, transport_network):
        """Congestion cost are perceived costs, felt by firms, but they do not influence 
        prices paid to transporter, hence do not change price
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

    def add_congestion_malus(self, graph, transport_network):
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
                    # if it is, compare actual cost with normal cost
                    actual_route_time_cost = transport_network.giveRouteCostWithCongestion(route_to_check)

                    # If it is on the main route, then there was no previous price increase due to transport
                    if graph[self][edge[1]]['object'].current_route == 'main':
                        relative_cost_change_no_congestion = 0
                        relative_cost_change_with_congestion = (actual_route_time_cost - graph[self][edge[1]][
                            'object'].route_time_cost) / graph[self][edge[1]]['object'].route_time_cost
                        self.finance['costs']['transport'] += (self.eq_finance['costs']['transport'] *
                                                               self.clients[edge[1].pid]['share'] *
                                                               (1 + relative_cost_change_with_congestion))

                    # Otherwise, we need to incremen the added price increase due to transport
                    elif graph[self][edge[1]]['object'].current_route == 'alternative':
                        relative_cost_change_no_congestion = (graph[self][edge[1]][
                                                                  'object'].alternative_route_time_cost -
                                                              graph[self][edge[1]]['object'].route_time_cost) / \
                                                             graph[self][edge[1]]['object'].route_time_cost
                        relative_cost_change_with_congestion = (actual_route_time_cost - graph[self][edge[1]][
                            'object'].route_time_cost) / graph[self][edge[1]]['object'].route_time_cost

                    else:
                        route_type = graph[self][edge[1]]['object'].current_route
                        raise NotImplementedError(f"Route {route_type} not recognised")

                    # We increment financial costs
                    self.finance['costs']['transport'] += (self.eq_finance['costs']['transport']
                                                           * self.clients[edge[1].pid]['share'] * (
                                                                   1 + relative_cost_change_with_congestion
                                                                   - relative_cost_change_no_congestion))

                    # We compute the new price increase due to transport and congestion
                    relative_price_change_transport_no_congestion = (self.eq_finance['costs']['transport']
                                                                     * relative_cost_change_no_congestion / (
                                                                             (1 - self.target_margin) *
                                                                             self.eq_finance['sales']))
                    relative_price_change_transport_with_congestion = (self.eq_finance['costs']['transport']
                                                                       * relative_cost_change_with_congestion / (
                                                                               (1 - self.target_margin) *
                                                                               self.eq_finance['sales']))
                    total_relative_price_change = (self.delta_price_input
                                                   + relative_price_change_transport_with_congestion)
                    new_price = graph[self][edge[1]]['object'].eq_price * (1 + total_relative_price_change)
                    delta_p_congestion = (relative_price_change_transport_with_congestion
                                          - relative_price_change_transport_no_congestion)
                    logging_msg = f"Firm {self.pid}: price of transport to {edge[1].pid} is impacted by congestion." \
                                  f"New price {new_price} vs. {graph[self][edge[1]]['object'].price}." \
                                  f"Route cost with congestion: {actual_route_time_cost} vs." \
                                  f" normal: {graph[self][edge[1]]['object'].route_time_cost}," \
                                  f" delta input: {self.delta_price_input}, delta transport no congestion:" \
                                  f"{relative_price_change_transport_with_congestion}, delta congestion:" \
                                  f"{delta_p_congestion} "

                    logging.debug(logging_msg)
                    graph[self][edge[1]]['object'].price = new_price

                    # Retransfer shipment
                    transport_network.transport_shipment(graph[self][edge[1]]['object'])

    def receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network):
        agent_receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network)

    def evaluate_profit(self, graph):
        # Collect all payments received
        self.finance['sales'] = sum([
            graph[self][edge[1]]['object'].payment
            for edge in graph.out_edges(self)
        ])
        # Collect all payments made
        self.finance['costs']['input'] = sum([
            graph[edge[0]][self]['object'].payment
            for edge in graph.in_edges(self)
        ])
        # Compute profit
        self.profit = (self.finance['sales']
                       - self.finance['costs']['input']
                       - self.finance['costs']['other']
                       - self.finance['costs']['transport'])
        # Compute Margins
        expected_gross_margin_no_transport = 1 - sum(list(self.input_mix.values()))
        if self.finance['sales'] > 0:
            realized_gross_margin_no_transport = ((self.finance['sales'] - self.finance['costs']['input'])
                                                  / self.finance['sales'])
            realized_margin = self.profit / self.finance['sales']
        else:
            realized_gross_margin_no_transport = 0
            realized_margin = 0

        # Log discrepancies
        if abs(realized_gross_margin_no_transport - expected_gross_margin_no_transport) > 1e-6:
            logging.debug('Firm ' + str(self.pid) + ': realized gross margin without transport is ' +
                          '{:.3f}'.format(realized_gross_margin_no_transport) + " instead of " +
                          '{:.3f}'.format(expected_gross_margin_no_transport))

        if abs(realized_margin - self.target_margin) > 1e-6:
            logging.debug('Firm ' + str(self.pid) + ': my margin differs from the target one: ' +
                          '{:.3f}'.format(realized_margin) + ' instead of ' + str(self.target_margin))

    def print_info(self):
        print("\nFirm " + str(self.pid) + " from sector " + str(self.sector) + ":")
        print("suppliers:", self.suppliers)
        print("clients:", self.clients)
        print("input_mix:", self.input_mix)
        print("order_book:", self.order_book, "; total_order:", self.total_order)
        print("input_needs:", self.input_needs)
        print("purchase_plan:", self.purchase_plan)
        print("inventory:", self.inventory)
        print("production:", self.production, "; production target:", self.production_target, "; product stock:",
              self.product_stock)
        print("profit:", self.profit, ";", self.finance)
