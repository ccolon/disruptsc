class Route(object):

    def __init__(self, cost_per_ton: float):
        self.cost_per_ton = cost_per_ton
        self.transport_modes = None
        self.transport_nodes = None
        self.transport_edges = None

    @classmethod
    def from_node_list(cls, node_list: list):
        transport_infra_id_list = [[(node_list[0],)]] + \
                                  [[(node_list[i], node_list[i + 1]), (node_list[i + 1],)]
                                   for i in range(0, len(node_list) - 1)]
        transport_infra_id_list = [item for item_tuple in transport_infra_id_list for item in item_tuple]
        return cls(transport_infra_id_list)

    def extract_transport_edges(self):
        return [item for item in self if len(item) == 2]

    def extract_transport_nodes(self):
        return [item for item in self if len(item) == 1]

    def calculate_route_cost_per_ton(self):
        pass
