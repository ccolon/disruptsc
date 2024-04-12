from typing import TYPE_CHECKING

import numpy as np

from code.model.basic_functions import calculate_distance_between_agents, rescale_values, generate_weights
from code.agents.agent import determine_nb_suppliers, agent_receive_products_and_pay

import logging

from code.agents.agent import Agent, AgentList
from code.network.commercial_link import CommercialLink
from code.network.mrio import import_label

if TYPE_CHECKING:
    from code.network.sc_network import ScNetwork
    from code.agents.firm import FirmList
    from code.agents.country import CountryList


class Household(Agent):

    def __init__(self, pid, odpoint, name, long, lat, sector_consumption):
        super().__init__(
            agent_type="household",
            name=name,
            pid=pid,
            odpoint=odpoint,
            long=long,
            lat=lat
        )
        # Parameters depending on data
        self.sector_consumption = sector_consumption
        # Parameters depending on network
        self.purchase_plan = {}
        self.retailers = {}
        # Variables reset and updated at each time step
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.consumption_per_sector = {}
        self.consumption_loss_per_sector = {}
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.spending_per_sector = {}
        self.extra_spending_per_sector = {}
        # Cumulated variables reset at beginning and updated at each time step
        self.consumption_loss = 0
        self.extra_spending = 0

    def reset_variables(self):
        self.consumption_per_retailer = {}
        self.tot_consumption = 0
        self.spending_per_retailer = {}
        self.tot_spending = 0
        self.extra_spending = 0
        self.consumption_loss = 0
        self.extra_spending_per_sector = {}
        self.consumption_loss_per_sector = {}

    def initialize_var_on_purchase_plan(self):
        if len(self.purchase_plan) == 0:
            logging.warning("Households initialize variables based on purchase plan, but it is empty.")

        self.consumption_per_retailer = self.purchase_plan
        self.tot_consumption = sum(list(self.purchase_plan.values()))
        self.consumption_loss_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}
        self.spending_per_retailer = self.consumption_per_retailer
        self.tot_spending = self.tot_consumption
        self.extra_spending_per_sector = {sector: 0 for sector in self.purchase_plan.keys()}

    def identify_suppliers(self, sector: str, firm_list, country_list,
                           nb_suppliers_per_input: float, weight_localization: float, force_local: bool,
                           firm_data_type: str):
        if firm_data_type == "mrio":
            # if len(sector_id) == 3:  # case of countries
            if import_label in sector:  # case of countries
                supplier_type = "country"
                selected_supplier_ids = [sector[:3]]  # for countries, the id is extracted from the name
                supplier_weights = [1]

            else:  # case of firms
                supplier_type = "firm"
                potential_supplier_pids = [firm.pid for firm in firm_list if firm.sector == sector]
                if len(potential_supplier_pids) == 0:
                    raise ValueError(f"{self.id_str().capitalize()}: there should be one supplier for {sector}")
                # Choose based on importance
                prob_to_be_selected = np.array(rescale_values([firm_list[firm_pid].importance for firm_pid in
                                                               potential_supplier_pids]))
                prob_to_be_selected /= prob_to_be_selected.sum()
                selected_supplier_ids = np.random.choice(potential_supplier_pids,
                                                         p=prob_to_be_selected, size=1,
                                                         replace=False).tolist()
                supplier_weights = [1]

        else:
            if sector == "IMP":
                supplier_type = "country"
                # Identify countries as suppliers if the corresponding sector does export
                importance_threshold = 1e-6
                potential_suppliers = [
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
                supplier_type = "firm"
                potential_suppliers = [firm.pid for firm in firm_list if firm.sector == sector]
                if len(potential_suppliers) == 0:
                    raise ValueError(f"{self.id_str().capitalize()}: no supplier for input {sector}")
                if force_local:
                    potential_local_suppliers = [firm_id for firm_id in potential_suppliers
                                                 if firm_list[firm_id].odpoint == self.odpoint]
                    if len(potential_local_suppliers) > 0:
                        potential_suppliers = potential_local_suppliers
                    else:
                        pass
                        # logging.debug(f"{self.id_str().capitalize()}: no local supplier for input {sector_id}")
                distance_to_each = rescale_values([
                    calculate_distance_between_agents(self, firm_list[firm_id])
                    for firm_id in potential_suppliers
                ])

                importance_of_each = rescale_values([firm_list[firm_id].importance for firm_id in potential_suppliers])

                prob_to_be_selected = np.array(importance_of_each) / (np.array(distance_to_each) ** weight_localization)
                prob_to_be_selected /= prob_to_be_selected.sum()

            # Determine the number of supplier(s) to select. 1 or 2.
            nb_suppliers_to_choose = determine_nb_suppliers(nb_suppliers_per_input)

            # Select the supplier(s). It there is 2 suppliers, then we generate
            # random weight. It determines how much is bought from each supplier.
            selected_supplier_ids = np.random.choice(potential_suppliers,
                                                     p=prob_to_be_selected, size=nb_suppliers_to_choose,
                                                     replace=False).tolist()
            index_map = {supplier_id: position for position, supplier_id in enumerate(potential_suppliers)}
            selected_positions = [index_map[supplier_id] for supplier_id in selected_supplier_ids]
            selected_prob = [prob_to_be_selected[position] for position in selected_positions]
            supplier_weights = generate_weights(nb_suppliers_to_choose, selected_prob)

        return supplier_type, selected_supplier_ids, supplier_weights

    def select_suppliers(self, graph: "ScNetwork", firm_list: "FirmList", country_list: "CountryList",
                         nb_retailers: float, force_local: bool,
                         weight_localization: float, firm_data_type: str):
        print(f"{self.id_str()}: consumption {self.sector_consumption}")
        for sector, amount in self.sector_consumption.items():
            supplier_type, retailers, retailer_weights = self.identify_suppliers(sector, firm_list, country_list,
                                                                                 nb_retailers,
                                                                                 weight_localization,
                                                                                 force_local,
                                                                                 firm_data_type)

            # For each of them, create commercial link
            for retailer_id in retailers:
                # Retrieve the appropriate supplier object from the id
                # If it is a country we get it from the country list
                # If it is a firm we get it from the firm list
                if supplier_type == "country":
                    supplier_object = [country for country in country_list if country.pid == retailer_id][0]
                    link_category = 'import_B2C'
                    product_type = "imports"
                else:
                    supplier_object = firm_list[retailer_id]
                    link_category = 'domestic_B2C'
                    product_type = firm_list[retailer_id].sector_type

                # For each retailer, create an edge in the economic network
                graph.add_edge(supplier_object, self,
                               object=CommercialLink(
                                   pid=str(retailer_id) + '->' + str(self.pid),
                                   product=sector,
                                   product_type=product_type,
                                   category=link_category,
                                   supplier_id=retailer_id,
                                   buyer_id=self.pid)
                               )
                # Associate a weight in the commercial link, the household's purchase plan & retailer list, in the retailer's client list
                weight = retailer_weights.pop()
                graph[supplier_object][self]['weight'] = weight
                self.purchase_plan[retailer_id] = weight * self.sector_consumption[sector]
                self.retailers[retailer_id] = {'sector': sector, 'weight': weight}
                distance = calculate_distance_between_agents(self, supplier_object)
                supplier_object.clients[self.pid] = {
                    'sector': "households", 'share': 0, 'transport_share': 0, "distance": distance
                }  # The share of sales cannot be calculated now.

    def send_purchase_orders(self, graph):
        for edge in graph.in_edges(self):
            try:
                quantity_to_buy = self.purchase_plan[edge[0].pid]
            except KeyError:
                print("Households: No purchase plan for supplier", edge[0].pid)
                quantity_to_buy = 0
            graph[edge[0]][self]['object'].order = quantity_to_buy

    def receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network):
        agent_receive_products_and_pay(self, graph, transport_network, sectors_no_transport_network)

    def select_supplier_from_list(self, firm_list: "FirmList",
                                  nb_suppliers_to_choose: int, potential_firm_ids: list,
                                  distance: bool, importance: bool, weight_localization: float,
                                  force_same_odpoint=False):
        # reduce firm to choose to local ones
        if force_same_odpoint:
            same_odpoint_firms = [
                firm_id
                for firm_id in potential_firm_ids
                if firm_list[firm_id].odpoint == self.odpoint
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
                calculate_distance_between_agents(self, firm_list[firm_id])
                for firm_id in potential_firm_ids
            ])
            distance_weight = 1 / (np.array(distance_to_each) ** weight_localization)
        else:
            distance_weight = np.ones(len(potential_firm_ids))

        # importance weight
        if importance:
            importance_of_each = rescale_values([firm_list[firm_id].importance for firm_id in potential_firm_ids])
            importance_weight = np.array(importance_of_each)
        else:
            importance_weight = np.ones(len(potential_firm_ids))

        # create weight vector based on choice
        prob_to_be_selected = distance_weight * importance_weight
        prob_to_be_selected /= prob_to_be_selected.sum()

        # perform the random choice
        selected_supplier_ids = np.random.choice(
            potential_firm_ids,
            p=prob_to_be_selected,
            size=nb_suppliers_to_choose,
            replace=False
        ).tolist()
        # Choose weight if there are multiple suppliers
        index_map = {supplier_id: position for position, supplier_id in enumerate(potential_firm_ids)}
        selected_positions = [index_map[supplier_id] for supplier_id in selected_supplier_ids]
        selected_prob = [prob_to_be_selected[position] for position in selected_positions]
        supplier_weights = generate_weights(nb_suppliers_to_choose, importance_of_each=selected_prob)

        # return
        return selected_supplier_ids, supplier_weights


class HouseholdList(AgentList):
    # def __init__(self, household_list: list[Household]):
    #     super().__init__(household for household in household_list if isinstance(household, Household))
    pass
