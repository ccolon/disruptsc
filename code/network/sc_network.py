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


def add_or_append_to_dict(dictionary, key, value_to_add):
    if key in dictionary.keys():
        dictionary[key] += value_to_add
    else:
        dictionary[key] = value_to_add