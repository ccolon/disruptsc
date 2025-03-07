from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from disruptsc.network.transport_network import TransportNetwork


class Route(list):
    def __init__(self, node_list: list, transport_network: "TransportNetwork", shipment_method: str):
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
        self.cost_per_ton = sum([transport_network[source][target]['cost_per_ton_' + shipment_method]
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

    def revert(self):
        """
        Reverse the route in place. This method reverses the order
        of the route’s list representation as well as all other list-based properties.
        For any edge_attr (a tuple of length 2), the tuple is also reversed.
        """
        # Reverse the main list (which is the Route itself) while flipping edge_attr tuples.
        reversed_nodes_and_edges = []
        for item in reversed(self):
            if isinstance(item, tuple) and len(item) == 2:
                # For an edge_attr tuple, swap the two nodes.
                reversed_nodes_and_edges.append((item[1], item[0]))
            else:
                # For a node (singleton tuple) leave it unchanged.
                reversed_nodes_and_edges.append(item)

        self[:] = reversed_nodes_and_edges  # Update the list (i.e. self) with the reversed nodes and edges
        self.transport_nodes_and_edges = reversed_nodes_and_edges  # Update the property that holds the nodes and edges
        self.transport_nodes = list(reversed(self.transport_nodes))  # Reverse the list of nodes
        self.transport_edges = [(edge[1], edge[0]) for edge in
                                reversed(self.transport_edges)]  # Reverse the list of edges, flipping each edge_attr tuple
        self.transport_edge_ids.reverse()  # Reverse the list of edge_attr IDs
        # The modes, cost per ton and length do not need to be reversed, as they are not order-dependent.
