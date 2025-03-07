from typing import TYPE_CHECKING

import logging

import geopandas
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

from disruptsc.network.transport_network import TransportNetwork

if TYPE_CHECKING:
    from pathlib import Path


def load_transport_data(filepaths, transport_mode, time_resolution):
    edges = gpd.read_file(filepaths[transport_mode + '_edges'])
    edges.index = edges['id']
    edges.index.name = 'index'
    edges['type'] = transport_mode

    # Compute how much it costs to transport one USD worth of good on each edge_attr
    # edges = compute_cost_travel_time_edges(edges, transport_params, transport_mode, transport_cost_data)

    # Adapt capacity (given in tons per day) to time resolution
    time_resolution_in_days = {'day': 1, 'week': 7, 'month': 365.25 / 12, 'year': 365.25}
    edges['capacity'] = pd.to_numeric(edges['capacity'], errors="coerce")
    edges['capacity'] = edges['capacity'] * time_resolution_in_days[time_resolution]

    # When there is no capacity, it means that there is no limitation
    unlimited_capacity = 1e9 * time_resolution_in_days[time_resolution]  # tons per year
    no_capacity_cond = (edges['capacity'].isnull()) | (edges['capacity'] == 0)
    edges.loc[no_capacity_cond, 'capacity'] = unlimited_capacity

    return edges


def offset_ids_legacy(nodes, edges, offset_node_id, offset_edge_id):
    # offset node id
    nodes['id'] = nodes['id'] + offset_node_id
    nodes.index = nodes['id']
    nodes.index.name = "index"
    # offset edge_attr id
    edges['id'] = edges['id'] + offset_edge_id
    edges.index = edges['id']
    edges.index.name = "index"
    # offset end1, end2
    edges['end1'] = edges['end1'] + offset_node_id
    edges['end2'] = edges['end2'] + offset_node_id

    return nodes, edges


def offset_ids(edges, offset_edge_id):
    # offset edge_attr id
    edges['id'] = edges['id'] + offset_edge_id
    edges.index = edges['id']
    edges.index.name = "index"
    return edges


def create_transport_network(transport_modes: list, filepaths: dict, transport_cost_data: dict,
                             logistics_parameters: dict, time_resolution: str):
    """Create the transport network object

    It uses one shapefile for the nodes and another for the edges.
    Note that there are strong constraints on these files, in particular on their attributes.
    We can optionally use an additional edge_attr shapefile, which contains extra road segments. Useful for scenario testing.

    Parameters
    ----------
    logistics_parameters
    time_resolution
    transport_cost_data
    transport_modes : list
        List of transport modes to include, ['roads', 'railways', 'waterways', 'airways']

    filepaths : dic
        Dic of filepaths

    Returns
    -------
    TransportNetwork, transport_nodes geopandas.DataFrame, transport_edges geopandas.DataFrame
    """
    # Create the transport network object
    logging.info('Creating transport network')

    # Load node and edge_attr data
    # Load in the following order: roads, railways, waterways, maritime, airways
    # Ids are adjusted to be unique
    # E.g., if roads and railways are included, the ids of railways nodes are offset such that
    # the first railway node id is the last road node id + 1
    # similarly for edges
    # the nodes ids in "end1", "end2" of edges are also offseted
    logging.info(f"Transport modes modeled: {transport_modes}")
    logging.info('Retrieving road data')
    edges = load_transport_data(filepaths, transport_mode="roads", time_resolution=time_resolution)
    for transport_mode in set(transport_modes) - {"roads"}:
        logging.info(f'Retrieving {transport_mode} data')
        edges = add_transport_mode(transport_mode, edges, filepaths, time_resolution)

    if len(transport_modes) >= 2:
        logging.debug('Retrieving multimodal data')
        multimodal_edges = load_transport_data(filepaths, "multimodal", time_resolution=time_resolution)
        multimodal_edges = select_multimodal_edges_needed(multimodal_edges, transport_modes)
        logging.debug(str(multimodal_edges.shape[0]) + " multimodal edges")
        # multimodal_edges = assign_endpoints(multimodal_edges, nodes)
        multimodal_edges = offset_ids(multimodal_edges, offset_edge_id=edges['id'].max() + 1)
        edges = pd.concat([edges, multimodal_edges], ignore_index=False,
                          verify_integrity=True, sort=False)
        # logging.debug('Total nb of transport nodes: ' + str(nodes.shape[0]))
        logging.debug('Total nb of transport edges: ' + str(edges.shape[0]))

    # Check conformity
    # if nodes['id'].duplicated().sum() > 0:
    #     raise ValueError('The following node ids are duplicated: ' +
    #                      nodes.loc[nodes['id'].duplicated(), "id"])
    if edges['id'].duplicated().sum() > 0:
        raise ValueError(f"The following edge_attr ids are duplicated: {edges.loc[edges['id'].duplicated(), 'id']}")
    # edge_ends = set(edges['end1'].tolist() + edges['end2'].tolist())
    # edge_ends_not_in_node_data = edge_ends - set(nodes['id'])
    # if len(edge_ends_not_in_node_data) > 0:
    #     raise KeyError("The following node ids are given as 'end1' or 'end2' in edge_attr data " + \
    #                    "but are not in the node data: " + str(edge_ends_not_in_node_data))

    # Load the nodes and edges on the transport network object
    logging.info(f'Identifying endpoints')
    nodes, edges = create_nodes_and_update_edges(edges)

    # check all attributed are there
    logging.info(f'Initializing attributes endpoints')
    required_attributes = ['id', "type", 'surface', "geometry", "class", "km", 'special', "name",
                           "capacity", "disruption",
                           "cost_per_ton", "travel_time", "time_cost", 'cost_travel_time', 'cost_variability',
                           'agg_cost', "multimodes"]
    missing_attribute = list(set(required_attributes) - set(edges.columns))
    edges[missing_attribute] = None
    edges['node_tuple'] = list(zip(edges['end1'], edges['end2']))
    edges['shipments'] = [{} for _ in range(len(edges))]
    edges['disruption_duration'] = 0
    edges['current_load'] = 0
    edges['overused'] = False
    edges['current_capacity'] = edges['capacity']

    logging.info('Creating the network')
    transport_network = TransportNetwork()
    for _, edge in edges.iterrows():
        transport_network.add_edge(edge["end1"], edge["end2"], **edge)
    nodes['disruption_duration'] = 0
    nodes['shipments'] = [{} for _ in range(len(nodes))]
    nodes['firms_there'] = [[] for _ in range(len(nodes))]
    nodes['households_there'] = None
    selected_node_attributes = ['long', 'lat', 'disruption_duration', 'shipments', 'firms_there', 'households_there']
    nx.set_node_attributes(transport_network, nodes[selected_node_attributes].to_dict("index"))
    min_basic_cost = logistics_parameters['basic_cost']['maritime']
    min_time_cost = 1.0 / logistics_parameters['speeds']['maritime'] * logistics_parameters['cost_of_time']
    transport_network.min_cost_per_tonkm = min_basic_cost + min_time_cost

    transport_network.ingest_logistic_data(logistics_parameters)

    return transport_network, edges, nodes


def create_nodes_and_update_edges(edges: geopandas.GeoDataFrame):
    # create nodes from endpoints
    endpoints = geopandas.GeoDataFrame({"end1": edges.geometry.apply(lambda line: Point(line.coords[0])),
                                        "end2": edges.geometry.apply(lambda line: Point(line.coords[-1]))})
    all_endpoints = geopandas.GeoDataFrame(pd.concat([endpoints['end1'], endpoints['end2']]), columns=["geometry"],
                                           crs=edges.crs)
    all_endpoints['geometry_wkt'] = all_endpoints['geometry'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    nodes = all_endpoints.drop_duplicates('geometry_wkt').copy()
    nodes['id'] = range(nodes.shape[0])
    nodes.index = nodes['id']
    nodes['long'] = nodes['geometry'].x
    nodes['lat'] = nodes['geometry'].y

    # add nodes_id into end1 and end2 columns of edges
    end1_wkt = endpoints['end1'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    end2_wkt = endpoints['end2'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    edges['end1'] = end1_wkt.map(nodes.set_index('geometry_wkt')['id'])
    edges['end2'] = end2_wkt.map(nodes.set_index('geometry_wkt')['id'])

    return nodes, edges


def add_transport_mode(
        mode: str,
        edges: geopandas.GeoDataFrame,
        filepaths: dict,
        time_resolution: str
):
    logging.debug(f'Loading {mode} data')
    new_mode_edges = load_transport_data(filepaths, mode, time_resolution=time_resolution)
    logging.debug(f"{new_mode_edges.shape[0]} {mode} edges")
    new_mode_edges = offset_ids(new_mode_edges, offset_edge_id=edges['id'].max() + 1)
    edges = pd.concat([edges, new_mode_edges], ignore_index=False,
                      verify_integrity=True, sort=False)
    return edges


def select_multimodal_edges_needed(multimodal_edges, transport_modes):
    # Select multimodal edges between the specified transport modes
    modes = multimodal_edges['multimodes'].str.split('-', expand=True)
    boolean = modes.iloc[:, 0].isin(transport_modes) & modes.iloc[:, 1].isin(transport_modes)
    return multimodal_edges[boolean]


def assign_endpoints_one_edge(row, df_nodes):
    p1, p2 = get_endpoints_from_line(row['geometry'])
    id_closest_point1 = get_index_closest_point(p1, df_nodes)
    id_closest_point2 = get_index_closest_point(p2, df_nodes)
    row['end1'] = id_closest_point1
    row['end2'] = id_closest_point2
    return row


def assign_endpoints(df_links, df_nodes):
    return df_links.apply(lambda row: assign_endpoints_one_edge(row, df_nodes), axis=1)


def get_endpoints_from_line(linestring_obj):
    end1_coord = linestring_obj.coords[0]
    end2_coord = linestring_obj.coords[-1]
    return Point(*end1_coord), Point(*end2_coord)


def get_index_closest_point(point, df_with_points):
    distance_list = [point.distance(item) for item in df_with_points['geometry'].tolist()]
    return int(df_with_points.index[distance_list.index(min(distance_list))])
