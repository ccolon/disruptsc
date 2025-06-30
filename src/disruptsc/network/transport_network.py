import copy
import math
from typing import TYPE_CHECKING

import geopandas
import networkx as nx
import numpy as np
import pandas as pd
import logging

from disruptsc.model.basic_functions import add_or_append_to_dict
from disruptsc.network.route import Route
from disruptsc.parameters import TRANSPORT_MALUS

if TYPE_CHECKING:
    from disruptsc.network.commercial_link import CommercialLink


def degrees_to_km(lon1, lat1, lon2, lat2):
    lat_km = 111 * abs(lat2 - lat1)
    lon_km = 111 * abs(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(lat_km ** 2 + lon_km ** 2)


class TransportNetwork(nx.Graph):
    def __init__(self, graph=None, **attr):
        super().__init__(graph, **attr)
        self.shipment_methods = None
        self.min_cost_per_tonkm = None
        # self.cost_heuristic = None
        # manager = Manager()
        self.shortest_path_library = {'normal': {}, 'alternative': {}}
        # self.lock = threading.Lock()
        
        # Phase 1: Distance cache for od_points
        self._distance_cache = {}  # (node_id1, node_id2) -> distance_km

    def info(self):
        transport_modes = self.get_transport_modes()
        return f"Transport network with {len(transport_modes)} modes: {transport_modes}\n" \
               f"Nb of nodes: {len(self.nodes)}, nb of edges: {len(self.edges)}"

    def get_distance_between_nodes(self, node_id1: int, node_id2: int) -> float:
        """
        Get cached distance between two transport nodes.
        
        Parameters
        ----------
        node_id1 : int
            First transport node ID
        node_id2 : int  
            Second transport node ID
            
        Returns
        -------
        float
            Distance in kilometers between the nodes
        """
        if node_id1 == node_id2:
            return 0.0
            
        # Ensure consistent cache key ordering
        key = (min(node_id1, node_id2), max(node_id1, node_id2))
        
        if key not in self._distance_cache:
            # Get coordinates from network nodes
            node1_data = self._node[node_id1]
            node2_data = self._node[node_id2]
            
            # Calculate distance using existing function
            distance = degrees_to_km(
                node1_data['long'], node1_data['lat'],
                node2_data['long'], node2_data['lat']
            )
            self._distance_cache[key] = distance
            
        return self._distance_cache[key]
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the distance cache."""
        total_possible_pairs = len(self.nodes) * (len(self.nodes) - 1) // 2
        return {
            'cached_pairs': len(self._distance_cache),
            'total_possible_pairs': total_possible_pairs,
            'cache_hit_ratio': len(self._distance_cache) / max(1, total_possible_pairs),
            'memory_usage_kb': len(self._distance_cache) * 8 / 1024  # Rough estimate
        }

    def add_transport_node(self, node_id, all_nodes_data):  # used in add_transport_edge_with_nodes
        node_attributes = ["id", "geometry"]
        if "special" in all_nodes_data.columns:
            node_attributes += ['special']
        if "name" in all_nodes_data.columns:
            node_attributes += ['name']
        node_data = all_nodes_data.loc[node_id, node_attributes].to_dict()
        node_data['long'] = all_nodes_data.loc[node_id, "geometry"].x
        node_data['lat'] = all_nodes_data.loc[node_id, "geometry"].y
        node_data['shipments'] = {}
        node_data['disruption_duration'] = 0
        node_data['firms_there'] = []
        node_data['households_there'] = None
        node_data['type'] = 'road'
        self.add_node(node_id, **node_data)

    def get_transport_modes(self):
        return list(set(nx.get_edge_attributes(self, "type").values()))

    def log_km_per_transport_modes(self):
        km_per_mode = pd.DataFrame({
            "km": nx.get_edge_attributes(self, "km"),
            "type": nx.get_edge_attributes(self, "type")
        })
        km_per_mode = km_per_mode.groupby('type')['km'].sum().to_dict()
        logging.info("Total length of transport network is: " +
                     "{:.0f} km".format(sum(km_per_mode.values())))
        for mode, km in km_per_mode.items():
            logging.info(mode + ": {:.0f} km".format(km))
        logging.info(f'Nb of nodes: {len(self.nodes)}, nb of edges: {len(self.edges)}')

    # def add_transport_edge_with_nodes(self, edge_id: int,
    #                                   all_edges_data: geopandas.GeoDataFrame,
    #                                   all_nodes_data: geopandas.GeoDataFrame):
    #     # Selecting data
    #     edge_attributes = ['id', "type", 'surface', "geometry", "class", "km", 'special', "name",
    #                        "capacity", "disruption"]
    #     if all_edges_data['type'].nunique() > 1:  # if there are multiple modes
    #         edge_attributes += ['multimodes']
    #     edge_data = all_edges_data.loc[edge_id, edge_attributes].to_dict()
    #     end_ids = all_edges_data.loc[edge_id, ["end1", "end2"]].tolist()
    #     # Creating the start and end nodes
    #     if end_ids[0] not in self.nodes:
    #         self.add_transport_node(end_ids[0], all_nodes_data)
    #     if end_ids[1] not in self.nodes:
    #         self.add_transport_node(end_ids[1], all_nodes_data)
    #     # Creating the edge_attr
    #     self.add_edge(end_ids[0], end_ids[1], **edge_data)
    #     # print("edge_attr id:", edge_id, "| end1:", end_ids[0], "| end2:", end_ids[1], "| nb edges:", len(self.edges))
    #     # print(self.edges)
    #     edge = self[end_ids[0]][end_ids[1]]
    #     edge['node_tuple'] = (end_ids[0], end_ids[1])
    #     edge['shipments'] = {}
    #     edge['disruption_duration'] = 0
    #     edge['current_load'] = 0
    #     edge['overused'] = False
    #     edge['current_capacity'] = edge['capacity']

    def locate_firms_on_nodes(self, firms):
        """The nodes of the transport network stores the list of firms located there
        using the attribute "firms_there".
        There can be several firms in one node.
        "transport_nodes" is a geodataframe of the nodes. It also contains this list in the columns
        "firm_there" as a comma-separated string

        This function reinitialize those fields and repopulate them with the adequate information
        """
        # Reinitialize
        # transport_nodes['firms_there'] = ""
        for node_id in self.nodes:
            self._node[node_id]['firms_there'] = []
        # Locate firms
        for firm in firms.values():
            self._node[firm.od_point]['firms_there'].append(firm.pid)
            # transport_nodes.loc[transport_nodes['id'] == firm.od_point, "firms_there"] += (',' + str(firm.pid))

    def locate_households_on_nodes(self, households):
        """The nodes of the transport network stores the list of households located there
        using the attribute "household_there".
        There can only be one household in one node.
        "transport_nodes" is a geodataframe of the nodes. It also contains the id of the household.

        This function reinitialize those fields and repopulate them with the adequate information
        """
        # Reinitialize
        # transport_nodes['household_there'] = None
        for pid, household in households.items():
            self._node[household.od_point]['household_there'] = pid
            # transport_nodes.loc[transport_nodes['id'] == household.od_point, "household_there"] = pid

    def provide_shortest_route(self, origin_node: int, destination_node: int,
                               shipment_method: str, route_weight: str) -> Route or None:
        """nx.shortest_path returns path as list of nodes
        we transform it into a route, which contains nodes and edges:
        [(1,), (1,5), (5,), (5,8), (8,)]
        """
        if origin_node not in self.nodes:
            logging.info(f"Origin node {origin_node} not in the available transport network")
            return None

        elif destination_node not in self.nodes:
            logging.info(f"Destination node {destination_node} not in the available transport network")
            return None

        else:
            weight = route_weight + '_' + shipment_method
            try:
                sp = nx.shortest_path(self, origin_node, destination_node, weight=weight)
                # sp = nx.astar_path(self, origin_node, destination_node, weight=weight,
                #                    heuristic=self.cost_heuristic, cutoff=TRANSPORT_MALUS)
                route = Route(sp, self, shipment_method)
                # if route.is_edge_in_route("turkmenbashi", self):
                #     logging.info(f"{weight}: {route.sum_indicator(self, weight)}")
                #     logging.info(self[8030][8038][weight])
                return route
            except nx.NetworkXNoPath:
                logging.info(f"There is no path between {origin_node} and {destination_node}")
                return None

    def cost_heuristic(self, u, v):
        distance = degrees_to_km(
            self._node[u]['long'], self._node[u]['lat'],
            self._node[v]['long'], self._node[v]['lat']
        )
        return distance * self.min_cost_per_tonkm

    def get_undisrupted_network(self):
        # available_nodes = [node for node in self.nodes if self._node[node]['disruption_duration'] == 0]
        # available_subgraph = self.subgraph(available_nodes)
        available_edges = [edge for edge in self.edges if self[edge[0]][edge[1]]['disruption_duration'] == 0]
        available_subgraph = self.edge_subgraph(available_edges)
        available_transport_network = TransportNetwork(available_subgraph)
        available_transport_network.min_cost_per_tonkm = self.min_cost_per_tonkm
        return available_transport_network

    def disrupt_roads(self, disruption):
        # Disrupting nodes
        for node_id in disruption['node']:
            logging.info('Road node ' + str(node_id) +
                         ' gets disrupted for ' + str(disruption['duration']) + ' time steps')
            self._node[node_id]['disruption_duration'] = disruption['duration']
        # Disrupting edges
        for edge in self.edges:
            if self[edge[0]][edge[1]]['type'] == 'virtual':
                continue
            else:
                if self[edge[0]][edge[1]]['id'] in disruption['edge_attr']:
                    logging.info('Road edge_attr ' + str(self[edge[0]][edge[1]]['id']) +
                                 ' gets disrupted for ' + str(disruption['duration']) + ' time steps')
                    self[edge[0]][edge[1]]['disruption_duration'] = disruption['duration']

    def disrupt_one_edge(self, edge, capacity_reduction: float, duration: int):
        logging.info(f"Road edge_attr {self[edge[0]][edge[1]]['id']} gets disrupted for {duration} time steps, "
                     f"capacity reduction is {capacity_reduction}")
        self[edge[0]][edge[1]]['disruption_duration'] = duration

    def update_road_disruption_state(self):
        """
        One time step is gone
        The remaining duration of disruption is decreased by 1
        """
        for node in self.nodes:
            if self._node[node]['disruption_duration'] > 0:
                self._node[node]['disruption_duration'] -= 1
        for edge in self.edges:
            if self[edge[0]][edge[1]]['disruption_duration'] > 0:
                self[edge[0]][edge[1]]['disruption_duration'] -= 1

    def transport_shipment(self, commercial_link: "CommercialLink", capacity_constraint: bool):
        # Select the route to transport the shipment: main or alternative
        if commercial_link.current_route == 'main':
            route_to_take = commercial_link.route
        elif commercial_link.current_route == 'alternative':
            route_to_take = commercial_link.alternative_route
        else:
            raise ValueError(f"No current route: {commercial_link.current_route}")

        shipment = {
                "supplier_id": commercial_link.supplier_id,
                "buyer_id": commercial_link.buyer_id,
                "origin_node": commercial_link.origin_node,
                "destination_node": commercial_link.destination_node,
                "quantity": commercial_link.delivery,
                "tons": commercial_link.delivery_in_tons,
                "product_type": commercial_link.product_type,
                "flow_category": commercial_link.category,
                "price": commercial_link.price
            }
        # Propagate the shipments on the edges
        for transport_edge in route_to_take.transport_edges:
            self[transport_edge[0]][transport_edge[1]]['shipments'][commercial_link.pid] = shipment
        # Add the shipment to the destination node
        self._node[commercial_link.destination_node]['shipments'][commercial_link.pid] = shipment
        # Propagate the load
        self.update_load_on_route(route_to_take, commercial_link.delivery_in_tons, capacity_constraint)

    def access_edge(self, edge):
        return self[edge[0]][edge[1]]

    def _get_cost_per_ton_attributes(self, with_capacity: bool = False):
        u, v, data = next(iter(self.edges(data=True)))  # Get first edge with attributes
        # Filter keys that start with 'cost_per_ton' and end with 'capacity'
        if with_capacity:
            return [key for key in data.keys() if key.startswith("cost_per_ton_with_capacity")]
        else:
            return [key for key in data.keys() if key.startswith("cost_per_ton")
                    and not key.startswith("cost_per_ton_with")]

    def update_load_on_route(self, route: "Route", load: float, capacity_constraint: bool):
        """Affect a load to a route

        The current_load attribute of each edge_attr in the route will be increased by the new load.
        A load is typically expressed in tons. If the current_load exceeds the capacity,
        then capacity_burden is added to the capacity_weight. This will prevent firms from choosing this route
        """
        # logging.info("Edge (2610, 2589): current_load "+str(self[2610][2589]['current_load']))
        capacity_burden = 1e10
        cost_per_ton_with_capacity_attributes = self._get_cost_per_ton_attributes(with_capacity=True)
        for u, v in route.transport_edges:
            edge = self[u][v]
            edge['current_load'] += load

            # Check if the edge_attr to be used is not over capacity already
            if capacity_constraint:
                if edge['overused']:
                    logging.warning(f"Edge {(u, v)} ({edge['type']}, {edge['name']}) is over capacity and got selected")
                else:
                    if edge['current_load'] > edge['capacity']:
                        logging.info(f"Edge {(u, v)} ({edge['type']}) has reached its capacity: "
                                     f"{edge['current_load']:.0f} / {edge['capacity']:.0f}")
                        edge['overused'] = True
                        for cost_per_ton_with_capacity_labels in cost_per_ton_with_capacity_attributes:
                            edge[cost_per_ton_with_capacity_labels] += capacity_burden

    def reset_loads(self):
        """
        Reset current_load to 0
        If an edge_attr was burdened due to capacity exceed, we remove the burden
        """
        cost_per_ton_labels = self._get_cost_per_ton_attributes(with_capacity=False)
        cost_per_ton_with_capacity_labels = self._get_cost_per_ton_attributes(with_capacity=True)
        for u, v in self.edges:
            edge = self[u][v]
            edge['current_load'] = 0
            edge['overused'] = False
            for i in range(len(cost_per_ton_labels)):
                edge[cost_per_ton_labels[i]] = edge[cost_per_ton_with_capacity_labels[i]]

    def remove_all_shipments(self):
        for u, v in self.edges:
            self[u][v]['shipments'] = {}

    def remove_shipment(self, commercial_link):
        """Look for the shipment corresponding to the commercial link
        in any edges and nodes of the main and alternative route,
        and remove it
        """
        route_to_take = commercial_link.get_current_route()
        for transport_edge in route_to_take.transport_edges:
            del self[transport_edge[0]][transport_edge[1]]['shipments'][commercial_link.pid]
        del self._node[commercial_link.destination_node]['shipments'][commercial_link.pid]

    def compute_flow_per_segment(self, time_step) -> list:
        """
        Calculate flows of each category and product for each transport edge_attr.

        We calculate total flows:
          - for each combination flow_category*product_type
          - for each flow_category
          - for each product_type
          - total of all
        """
        flows_per_edge = []
        # flows_total = {}

        for u, v in self.edges():
            edge_data = self[u][v]
            shipments = edge_data["shipments"].values()
            # km = edge_data["km"]

            new_data = {
                "time_step": time_step,
                "id": edge_data['id'],
                "flow_total": 0,
                "flow_total_tons": 0
            }

            for shipment in shipments:
                fc = shipment['flow_category']
                pt = shipment['product_type']
                qty = shipment['quantity']
                tons = shipment['tons']

                # Update new_data entries
                key_combo = f'flow_{fc}_{pt}'
                new_data[key_combo] = new_data.get(key_combo, 0) + qty
                key_fc = f'flow_{fc}'
                new_data[key_fc] = new_data.get(key_fc, 0) + qty
                key_pt = f'flow_{pt}'
                new_data[key_pt] = new_data.get(key_pt, 0) + qty
                new_data['flow_total'] += qty
                new_data['flow_total_tons'] += tons

                # Update flows_total entries
                # flows_total[fc] = flows_total.get(fc, 0) + qty
                # flows_total[f'{fc}*km'] = flows_total.get(f'{fc}*km', 0) + qty * km
                # flows_total[f'{fc}_tons'] = flows_total.get(f'{fc}_tons', 0) + tons
                # flows_total[f'{fc}_tons*km'] = flows_total.get(f'{fc}_tons*km', 0) + tons * km

            flows_per_edge.append(new_data)
        # logging.info(flows_total)
        return flows_per_edge

    def reinitialize_flows_and_disruptions(self):
        for node in self.nodes:
            node_data = self.nodes[node]
            node_data['disruption_duration'] = 0
            node_data['shipments'] = {}
        for u, v in self.edges():
            edge_data = self[u][v]
            edge_data['disruption_duration'] = 0
            edge_data['shipments'] = {}
            # self[edge_attr[0]][edge_attr[1]]['congestion'] = 0
            edge_data['current_load'] = 0
            edge_data['overused'] = False

    def ingest_logistic_data(self, logistic_parameters: dict):
        # Apply the function based on the edge_attr type
        self.shipment_methods = list(logistic_parameters['shipment_methods_to_transport_modes'].keys())
        nb_cost_profiles = logistic_parameters['nb_cost_profiles']
        self.shortest_path_library = {
            i: {
                'normal': {method: {} for method in self.shipment_methods},
                'alternative': {method: {} for method in self.shipment_methods}
            } for i in range(nb_cost_profiles)
        }
        for _, attr in self.edges.items():
            _calculate_cost_per_ton(attr, logistic_parameters)

    def retrieve_cached_route(self, from_node: int, to_node: int, cost_profile: int,
                              normal_or_disrupted: str, shipment_method: str):
        library_key = tuple(sorted((from_node, to_node)))
        canonical_route = self.shortest_path_library[cost_profile][normal_or_disrupted][shipment_method].get(library_key)
        if canonical_route:
            if from_node == library_key[0]:
                return canonical_route
            else:
                route = copy.deepcopy(canonical_route)
                route.revert()
                return route

    def cache_route(self, route: "Route", from_node: int, to_node: int, cost_profile: int,
                    normal_or_disrupted: str, shipment_method: str):
        library_key = tuple(sorted((from_node, to_node)))
        if from_node == library_key[0]:
            self.shortest_path_library[cost_profile][normal_or_disrupted][shipment_method][library_key] = route
        else:
            canonical_route = copy.deepcopy(route)
            canonical_route.revert()
            self.shortest_path_library[cost_profile][normal_or_disrupted][shipment_method][library_key] = canonical_route

    def check_no_uncollected_shipment(self):
        for u, v in self.edges:
            edge_data = self[u][v]
            if len(edge_data['shipments']) > 0:
                raise ValueError(f"There are uncollected shipments: {list(edge_data['shipments'].keys())}")

def _get_speed(edge_attr: dict, speed_dict: dict) -> float:
    if edge_attr['type'] == "roads":
        # if edge_attr['class'] == 'primary':
        #     return speed_dict['roads']['primary']
        # elif edge_attr['class'] == 'secondary':
        #     return speed_dict['roads']['primary']
        # elif edge_attr['class'] == 'tertiary':
        #     return speed_dict['roads']['primary']
        # else:
        if edge_attr['surface'] == 'paved':
            return speed_dict['roads']['paved']
        elif edge_attr['surface'] == 'unpaved':
            return speed_dict['roads']['unpaved']
        else:
            return speed_dict['roads']['paved']
    elif edge_attr['type'] in ['railways', 'waterways', 'maritime', 'airways', "pipelines"]:
        return speed_dict[edge_attr['type']]
    elif edge_attr['type'] == "multimodal":
        return speed_dict['roads']['paved']  # these are very small links, so we assume paved roads


def _get_loading_time_and_fee(edge_attr: dict, loading_times: dict, loading_fees: dict) -> (float, float):
    if edge_attr['type'] == "multimodal":
        return loading_times[edge_attr['multimodes']], loading_fees[edge_attr['multimodes']]
    else:
        return 0.0, 0.0


def _get_border_crossing_time_and_fee(edge_attr: dict, border_crossing_times: dict,
                                      border_crossing_fees: dict) -> (float, float):
    if isinstance(edge_attr['special'], str):
        if "custom" in edge_attr['special']:
            return border_crossing_times[edge_attr['type']], border_crossing_fees[edge_attr['type']]
    return 0.0, 0.0


def _calculate_cost_per_ton(edge_attr, logistic_parameters: dict):
    # calculate cost per ton
    basic_costs = {i: edge_attr['km'] * basic_cost_random[edge_attr['type']]
                   for i, basic_cost_random in logistic_parameters['basic_cost_profiles'].items()}
    transport_time = edge_attr['km'] / _get_speed(edge_attr, logistic_parameters['speeds'])
    loading_time, loading_fee = _get_loading_time_and_fee(
        edge_attr, logistic_parameters['loading_times'], logistic_parameters['loading_fees'])
    border_crossing_time, border_crossing_fee = _get_border_crossing_time_and_fee(
        edge_attr, logistic_parameters['border_crossing_times'], logistic_parameters['border_crossing_fees'])
    total_time = transport_time + loading_time + border_crossing_time
    total_fee = loading_fee + border_crossing_fee
    special_cost = logistic_parameters['name-specific'].get(edge_attr['name'], 0)
    costs_per_ton = {i: basic_cost + special_cost + total_fee + total_time * logistic_parameters['cost_of_time']
                     for i, basic_cost in basic_costs.items()}

    # add malus for non-supported shipment types
    shipment_type_malus = TRANSPORT_MALUS
    for shipment_method, transport_mode in logistic_parameters['shipment_methods_to_transport_modes'].items():
        for i, cost_per_ton in costs_per_ton.items():
            if edge_attr['type'] in transport_mode:
                edge_attr['cost_per_ton_' + str(i) + "_" + shipment_method] = cost_per_ton
                edge_attr['cost_per_ton_with_capacity_' + str(i) + "_" + shipment_method] = cost_per_ton
            else:
                edge_attr['cost_per_ton_' + str(i) + "_" + shipment_method] = shipment_type_malus
                edge_attr['cost_per_ton_with_capacity_' + str(i) + "_" + shipment_method] = shipment_type_malus
