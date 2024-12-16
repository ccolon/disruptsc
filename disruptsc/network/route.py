from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from disruptsc.network.transport_network import TransportNetwork


class Route(list):
    def __init__(self, node_list: list, transport_network: "TransportNetwork"):
        node_edge_tuple = [[(node_list[0],)]] + \
                          [[(node_list[i], node_list[i + 1]), (node_list[i + 1],)]
                           for i in range(0, len(node_list) - 1)]
        transport_nodes_and_edges = [item for item_tuple in node_edge_tuple for item in item_tuple]
        super().__init__(transport_nodes_and_edges)
        self.transport_nodes_and_edges = transport_nodes_and_edges
        self.transport_nodes = node_list
        self.transport_edges = [item for item in transport_nodes_and_edges if len(item) == 2]
        self.transport_edge_ids = [transport_network[source][target]['id'] for source, target in self.transport_edges]
        self.transport_modes = list(set([transport_network[source][target]['type']
                                         for source, target in self.transport_edges]))
        # self.transport_modes = list(transport_edges.loc[self.transport_edge_ids, 'type'].unique())
        self.cost_per_ton = sum([transport_network[source][target]['cost_per_ton']
                                 for source, target in self.transport_edges])
        # self.cost_per_ton = transport_edges.loc[self.transport_edge_ids, 'cost_per_ton'].sum()
        self.length = sum([transport_network[source][target]['km']
                           for source, target in self.transport_edges])

    def check_edge_in_route(self, route, searched_edge):
        for edge in self.transport_edges:
            if (searched_edge[0] == edge[0]) and (searched_edge[1] == edge[1]):
                return True
        return False

    def sum_indicator(self, transport_network: "TransportNetwork", indicator: str, per_type: bool = False):
        if per_type:
            details = []
            for edge in self.transport_edges:
                new_edge = {'id': transport_network[edge[0]][edge[1]]['id'],
                            'type': transport_network[edge[0]][edge[1]]['type'],
                            'multimodes': transport_network[edge[0]][edge[1]]['multimodes'],
                            'special': transport_network[edge[0]][edge[1]]['special'],
                            indicator: transport_network[edge[0]][edge[1]][indicator]}
                details += [new_edge]
            details = pd.DataFrame(details).fillna('N/A')
            # print(details)
            return details.groupby(['type', 'multimodes', 'special'])[indicator].sum()

        else:
            total_indicator = 0
            for edge in self.transport_edges:
                total_indicator += transport_network[edge[0]][edge[1]][indicator]
            return total_indicator
