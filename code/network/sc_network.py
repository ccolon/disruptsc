import networkx as nx
import pandas as pd


class ScNetwork(nx.DiGraph):

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
            elif commercial_link.category == "import":
                add_or_append_to_dict(io, ("IMP", buyer.sector), commercial_link.order)
            elif commercial_link.category == "transit":
                pass
            else:
                raise KeyError('Commercial link categories should be one of domestic_B2B, '
                               'domestic_B2C, export, import, transit')

        io_table = pd.Series(io).unstack().fillna(0)
        return io_table

    def generate_edge_list(self):
        edge_list = [(source.pid, source.agent_type, source.odpoint, target.pid, target.agent_type, target.odpoint)
                     for source, target in self.edges()]
        edge_list = pd.DataFrame(edge_list)
        edge_list.columns = ['source_id', 'source_type', 'source_od_point',
                             'target_id', 'target_type', 'target_od_point']
        return edge_list


def add_or_append_to_dict(dictionary, key, value_to_add):
    if key in dictionary.keys():
        dictionary[key] += value_to_add
    else:
        dictionary[key] = value_to_add
