import logging
from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd

from disruptsc.model.basic_functions import add_or_append_to_dict

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.agents.country import Countries
    from disruptsc.agents.household import Households


class ScNetwork(nx.DiGraph):

    def access_commercial_link(self, edge):
        return self[edge[0]][edge[1]]['object']

    def calculate_io_matrix(self):
        io = {}
        for supplier, buyer, data in self.edges(data=True):
            commercial_link = data['object']
            if commercial_link.category == "domestic_B2C":
                add_or_append_to_dict(io, (supplier.sector, 'final_demand'), commercial_link.order)
            elif commercial_link.category == "export":
                add_or_append_to_dict(io, (supplier.sector, 'export'), commercial_link.order)
            elif commercial_link.category == "domestic_B2B":
                add_or_append_to_dict(io, (supplier.sector, buyer.sector), commercial_link.order)
            elif commercial_link.category == "import_B2C":
                add_or_append_to_dict(io, ("IMP", 'final_demand'), commercial_link.order)
            elif commercial_link.category == "import":
                add_or_append_to_dict(io, ("IMP", buyer.sector), commercial_link.order)
            elif commercial_link.category == "transit":
                pass
            else:
                raise KeyError('Commercial link categories should be one of domestic_B2B, '
                               'domestic_B2C, export, import, import_B2C, transit')

        io_table = pd.Series(io).unstack().fillna(0)
        return io_table

    def generate_edge_list(self):
        edge_list = [(source.pid, source.id_str(), source.agent_type, source.od_point,
                      target.pid, target.id_str(), target.agent_type, target.od_point)
                     for source, target in self.edges()]
        edge_list = pd.DataFrame(edge_list)
        edge_list.columns = ['source_id', 'source_str_id', 'source_type', 'source_od_point',
                             'target_id', 'target_str_id', 'target_type', 'target_od_point']
        return edge_list

    def identify_firms_without_clients(self):
        return [node for node in self.nodes() if (self.out_degree(node) == 0) and node.agent_type == "firm"]

    def identify_disconnected_nodes(self, firms: "Firms", countries: "Countries", households: "Households"):
        firm_ids = list(firms.keys())
        country_ids = list(countries.keys())
        household_ids = list(households.keys())
        node_id_in_sc_network = [node.pid for node in self]
        disconnected_nodes = {}
        if len(set(firm_ids) - set(node_id_in_sc_network)) > 0:
            disconnected_nodes['firms'] = list(set(firm_ids) - set(node_id_in_sc_network))
        if len(set(country_ids) - set(node_id_in_sc_network)) > 0:
            disconnected_nodes['countries'] = list(set(country_ids) - set(node_id_in_sc_network))
        if len(set(household_ids) - set(node_id_in_sc_network)) > 0:
            disconnected_nodes['households'] = list(set(household_ids) - set(node_id_in_sc_network))
        return disconnected_nodes

    def remove_useless_commercial_links(self):
        firms_without_clients = self.identify_firms_without_clients()
        logging.info(f"Removing {len(firms_without_clients)} firms without clients and their associated links")
        
        for firm_without_clients in firms_without_clients:
            # Remove all incoming edges and update supplier records
            suppliers = [edge[0] for edge in self.in_edges(firm_without_clients)]
            for supplier in suppliers:
                self.remove_edge(supplier, firm_without_clients)
                if firm_without_clients.pid in supplier.clients:
                    del supplier.clients[firm_without_clients.pid]
                if supplier.pid in firm_without_clients.suppliers:
                    del firm_without_clients.suppliers[supplier.pid]
            
            # Remove the firm node from the supply chain network
            self.remove_node(firm_without_clients)
        
        return len(firms_without_clients)  # Return count for validation
