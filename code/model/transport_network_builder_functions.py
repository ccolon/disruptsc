import logging

import geopandas
import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point

from code.network.transport_network import TransportNetwork


def load_transport_data(filepaths, transport_params, transport_mode, additional_roads=None):
    # Determines whether there are nodes and/or edges to load
    any_edge = True
    any_node = True
    if transport_mode == "multimodal":
        any_node = False

    # Load nodes
    if any_node:
        nodes = gpd.read_file(filepaths[transport_mode + '_nodes'])
        # if 'index' in nodes.columns: #remove any column named "index". Seems to create issues
        #     nodes = nodes.drop('index', axis=1)
        nodes.index = nodes['id']
        nodes.index.name = 'index'
        nodes['type'] = transport_mode

    # Load edges
    if any_edge:
        edges = gpd.read_file(filepaths[transport_mode + '_edges'])
        # if 'index' in edges.columns: #remove any column named "index". Seems to create issues
        #     edges = edges.drop('index', axis=1)
        edges.index = edges['id']
        edges.index.name = 'index'
        edges['type'] = transport_mode

        # Add additional road edges, if any
        if (transport_mode == "roads") and additional_roads:
            offset_edge_id = edges['id'].max() + 1
            new_road_edges = gpd.read_file(filepaths['extra_roads_edges'])
            # if 'index' in new_road_edges.columns: #remove any column named "index". Seems to create issues
            #     new_road_edges = new_road_edges.drop('index', axis=1)
            new_road_edges['id'] = new_road_edges['id'] + offset_edge_id
            new_road_edges.index = new_road_edges['id']
            new_road_edges['type'] = 'road'
            edges = pd.concat([edges, new_road_edges], ignore_index=False,
                              verify_integrity=True, sort=False)

        # Compute how much it costs to transport one USD worth of good on each edge
        edges = compute_cost_travel_time_edges(edges, transport_params, edge_type=transport_mode)

        # Adapt capacity (given in year) to time resolution
        periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
        time_resolution = "week"
        edges['capacity'] = pd.to_numeric(edges['capacity'], errors="coerce")
        edges['capacity'] = edges['capacity'] / periods[time_resolution]

        # When there is no capacity, it means that there is no limitation
        unlimited_capacity = 1e9 * periods[time_resolution]  # tons per year
        no_capacity_cond = (edges['capacity'].isnull()) | (edges['capacity'] == 0)
        edges.loc[no_capacity_cond, 'capacity'] = unlimited_capacity
        # dic_capacity = {
        #     "roads": 1000000*52,
        #     "railways": 20000*52,
        #     "waterways": 40000*52,
        #     "maritime": 1e12*52,
        #     "multimodal": 1e12*52
        # }
        # edges['capacity'] = dic_capacity[transport_mode] / periods[time_resolution]
        # if (transport_mode == "roads"):
        #     edges.loc[edges['surface']=="unpaved", "capacity"] = 100000*52 / periods[time_resolution]

    # Return nodes, edges, or both
    if any_node and any_edge:
        return nodes, edges
    if any_node and ~any_edge:
        return nodes
    if ~any_node and any_edge:
        return edges


def offset_ids(nodes, edges, offset_node_id, offset_edge_id):
    # offset node id
    nodes['id'] = nodes['id'] + offset_node_id
    nodes.index = nodes['id']
    nodes.index.name = "index"
    # offset edge id
    edges['id'] = edges['id'] + offset_edge_id
    edges.index = edges['id']
    edges.index.name = "index"
    # offset end1, end2
    edges['end1'] = edges['end1'] + offset_node_id
    edges['end2'] = edges['end2'] + offset_node_id

    return nodes, edges


def create_transport_network(transport_modes: list, filepaths: dict, extra_roads=False):
    """Create the transport network object

    It uses one shapefile for the nodes and another for the edges.
    Note that there are strong constraints on these files, in particular on their attributes.
    We can optionally use an additional edge shapefile, which contains extra road segments. Useful for scenario testing.

    Parameters
    ----------
    extra_roads
    transport_modes : list
        List of transport modes to include, ['roads', 'railways', 'waterways', 'airways']

    filepaths : dic
        Dic of filepaths

    Returns
    -------
    TransportNetwork, transport_nodes geopandas.DataFrame, transport_edges geopandas.DataFrame
    """

    # Load transport parameters
    with open(filepaths['transport_parameters'], "r") as yaml_file:
        transport_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Create the transport network object
    logging.info('Creating transport network')
    if extra_roads:
        logging.info('Including extra roads')
    transport_network = TransportNetwork()
    # T.graph['unit_cost'] = transport_params['transport_cost_per_tonkm']

    # Load node and edge data
    # Load in the following order: roads, railways, waterways
    # Ids are adjusted to be unique
    # E.g., if roads and railways are included, the ids of railways nodes are offset such that
    # the first railway node id is the last road node id + 1
    # similarly for edges
    # the nodes ids in "end1", "end2" of edges are also offseted
    logging.debug('Loading roads data')
    nodes, edges = load_transport_data(filepaths, transport_params,
                                       transport_mode="roads", additional_roads=extra_roads)
    logging.info(f"Transport modes modeled: {transport_modes}")
    if "railways" in transport_modes:
        nodes, edges = add_transport_mode("railways", nodes, edges, filepaths, transport_params)

    if "waterways" in transport_modes:
        nodes, edges = add_transport_mode("waterways", nodes, edges, filepaths, transport_params)

    if "maritime" in transport_modes:
        nodes, edges = add_transport_mode("maritime", nodes, edges, filepaths, transport_params)

    if "airways" in transport_modes:
        nodes, edges = add_transport_mode("airways", nodes, edges, filepaths, transport_params)

    if len(transport_modes) >= 2:
        logging.debug('Loading multimodal data')
        multimodal_edges = load_transport_data(filepaths, transport_params, "multimodal")
        multimodal_edges = select_multimodal_edges_needed(multimodal_edges, transport_modes)
        logging.debug(str(multimodal_edges.shape[0]) + " multimodal edges")
        multimodal_edges = assign_endpoints(multimodal_edges, nodes)
        multimodal_edges['id'] = multimodal_edges['id'] + edges['id'].max() + 1
        multimodal_edges.index = multimodal_edges['id']
        multimodal_edges.index.name = "index"
        edges = pd.concat([edges, multimodal_edges], ignore_index=False,
                          verify_integrity=True, sort=False)
        logging.debug('Total nb of transport nodes: ' + str(nodes.shape[0]))
        logging.debug('Total nb of transport edges: ' + str(edges.shape[0]))

    # Check conformity
    if nodes['id'].duplicated().sum() > 0:
        raise ValueError('The following node ids are duplicated: ' +
                         nodes.loc[nodes['id'].duplicated(), "id"])
    if edges['id'].duplicated().sum() > 0:
        raise ValueError('The following edge ids are duplicated: ' +
                         edges.loc[edges['id'].duplicated(), "id"])
    edge_ends = set(edges['end1'].tolist() + edges['end2'].tolist())
    edge_ends_not_in_node_data = edge_ends - set(nodes['id'])
    if len(edge_ends_not_in_node_data) > 0:
        raise KeyError("The following node ids are given as 'end1' or 'end2' in edge data " + \
                       "but are not in the node data: " + str(edge_ends_not_in_node_data))

    # Load the nodes and edges on the transport network object
    logging.debug('Creating transport nodes and edges as a network')
    for road in edges['id']:
        transport_network.add_transport_edge_with_nodes(road, edges, nodes)

    return transport_network, nodes, edges


def add_transport_mode(
        mode: str,
        nodes: geopandas.GeoDataFrame,
        edges: geopandas.GeoDataFrame,
        filepaths: dict,
        transport_params: dict
):
    logging.debug(f'Loading {mode} data')
    new_mode_nodes, new_mode_edges = load_transport_data(filepaths, transport_params, mode)
    logging.debug(f"{new_mode_nodes.shape[0]} {mode} nodes and {new_mode_edges.shape[0]} {mode} edges")
    new_mode_nodes, new_mode_edges = offset_ids(new_mode_nodes, new_mode_edges,
                                                offset_node_id=nodes['id'].max() + 1,
                                                offset_edge_id=edges['id'].max() + 1)
    nodes = pd.concat([nodes, new_mode_nodes], ignore_index=False,
                      verify_integrity=True, sort=False)
    edges = pd.concat([edges, new_mode_edges], ignore_index=False,
                      verify_integrity=True, sort=False)
    return nodes, edges


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


def compute_cost_travel_time_edges(edges: geopandas.GeoDataFrame, transport_params: dict,
                                   edge_type: str) -> geopandas.GeoDataFrame:
    # A. Compute price per ton to be paid to transporter
    # A1. Cost per km
    if edge_type == "roads":
        # Differentiate between paved and unpaved roads
        edges['cost_per_ton'] = edges['km'] * (
                (edges['surface'] == 'unpaved') * transport_params['transport_cost_per_tonkm']['roads']['unpaved'] + \
                (edges['surface'] == 'paved') * transport_params['transport_cost_per_tonkm']['roads']['paved']
        )
    elif edge_type in ['railways', 'waterways', 'maritime', 'airways']:
        edges['cost_per_ton'] = edges['km'] * transport_params['transport_cost_per_tonkm'][edge_type]

    elif edge_type == "multimodal":
        edges['cost_per_ton'] = edges['multimodes'].map(transport_params['loading_cost_per_ton'])

    else:
        raise ValueError("'edge_type' should be 'roads', 'railways', 'waterways', 'airways', or 'multimodal'")

    # A2. Forwarding charges at borders
    if not edges['special'].isnull().all():
        boolean_custom_edge = edges['special'].str.contains("custom", na=False)
        edges.loc[boolean_custom_edge, "cost_per_ton"] = edges.loc[boolean_custom_edge, "type"] \
            .map(transport_params['custom_cost'])

    # B. Compute generalized cost of transport
    # B.1a. Compute travel time
    if edge_type == "roads":
        # Differentiate between paved and unpaved roads
        edges['travel_time'] = edges['km'] * (
                (edges['surface'] == 'unpaved') / transport_params['speeds']['roads']['unpaved'] +
                (edges['surface'] == 'paved') / transport_params['speeds']['roads']['paved']
        )
    elif edge_type in ['railways', 'waterways', 'maritime', 'airways']:
        edges['travel_time'] = edges['km'] / transport_params['speeds'][edge_type]

    elif edge_type == "multimodal":
        edges['travel_time'] = edges['km'] / edges['multimodes'].map(transport_params['loading_time'])

    else:
        raise ValueError("'edge_type' should be 'roads', 'railways', 'waterways', 'airways', or 'multimodal'")

    # B.1b. Add crossing border time
    if not edges['special'].isnull().all():
        boolean_custom_edge = edges['special'].str.contains("custom", na=False)
        edges.loc[boolean_custom_edge, "travel_time"] = edges.loc[boolean_custom_edge, "type"] \
                .map(transport_params['custom_time'])

    # B.2. Compute cost of travel time
    edges['cost_travel_time'] = edges['travel_time'] * transport_params["travel_cost_of_time"]

    # B.3. Compute variability cost
    if edge_type == "roads":
        # Differentiate between paved and unpaved roads
        edges['cost_variability'] = edges['cost_travel_time'] * \
            transport_params['variability_coef'] * (
                (edges['surface'] == 'unpaved') * transport_params["variability"]['roads']['unpaved']
                + (edges['surface'] == 'paved') * transport_params['variability']['roads']['paved']
            )

    elif edge_type in ['railways', 'waterways', 'maritime', 'airways']:
        edges['cost_variability'] = edges['cost_travel_time'] * \
                                    transport_params['variability_coef'] * \
                                    transport_params["variability"][edge_type]

    elif edge_type == "multimodal":
        edges['cost_variability'] = edges['cost_travel_time'] * \
                                    transport_params['variability_coef'] * \
                                    edges['multimodes'].map(transport_params['variability']['multimodal'])

    else:
        raise ValueError("'edge_type' should be 'roads', 'railways', 'waterways', 'airways', or 'multimodal'")

    # B.4. Finally, add up both cost to get the cost of time of each road edges
    edges['time_cost'] = edges['cost_travel_time'] + edges['cost_variability']

    # C. Compute an aggregate cost
    # It is "cost_per_ton" + "time_cost"
    edges['agg_cost'] = edges['time_cost'] / 2 + edges['cost_per_ton']
    return edges
