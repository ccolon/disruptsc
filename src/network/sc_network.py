import logging

import networkx as nx
import pandas as pd

from src.agents.firm import Firm
from src.model.basic_functions import add_or_append_to_dict


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
        edge_list = [(source.pid, source.agent_type, source.od_point, target.pid, target.agent_type, target.od_point)
                     for source, target in self.edges()]
        edge_list = pd.DataFrame(edge_list)
        edge_list.columns = ['source_id', 'source_type', 'source_od_point',
                             'target_id', 'target_type', 'target_od_point']
        return edge_list

    def identify_firms_without_clients(self):
        return [node for node in self.nodes() if (self.out_degree(node) == 0) and isinstance(node, Firm)]

    def remove_useless_commercial_links(self):
        firms_without_clients = self.identify_firms_without_clients()
        # print(firms_without_clients)
        logging.info(f"There are {len(firms_without_clients)} firms without clients. Removing associated links")
        for firm_without_clients in firms_without_clients:
            suppliers = [edge[0] for edge in self.in_edges(firm_without_clients)]
            for supplier in suppliers:
                self.remove_edge(supplier, firm_without_clients)
                del supplier.clients[firm_without_clients.pid]
                del firm_without_clients.suppliers[supplier.pid]
        # logging.info(f"There remain {len(self.identify_firms_without_clients())} firms without clients.")
        # print(self.identify_firms_without_clients())
