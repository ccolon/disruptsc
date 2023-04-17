import logging
import geopandas as gpd
import pandas as pd
import yaml


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
        edges = computeCostTravelTimeEdges(edges, transport_params, edge_type=transport_mode)

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


def create_transport_network(transport_modes, filepaths, extra_roads=False):
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
    T = TransportNetwork()
    # T.graph['unit_cost'] = transport_params['transport_cost_per_tonkm']

    # Load node and edge data
    # Load in the following order: roads, railways, waterways
    # Ids are adjusted to be unique
    # E.g., if roads and railways are included, the ids of railways nodes are offset such that
    # the first railway node id is the last road node id + 1
    # similarly for edges
    # the nodes ids in "end1", "end2" of edges are also offseted
    logging.debug('Loading roads data')
    nodes, edges = loadTransportData(filepaths, transport_params,
                                     transport_mode="roads", additional_roads=extra_roads)
    logging.debug(str(nodes.shape[0]) + " roads nodes and " +
                  str(edges.shape[0]) + " roads edges")

    if "railways" in transport_modes:
        logging.debug('Loading railways data')
        railways_nodes, railways_edges = loadTransportData(filepaths, transport_params, "railways")
        logging.debug(str(railways_nodes.shape[0]) + " railways nodes and " +
                      str(railways_edges.shape[0]) + " railways edges")
        railways_nodes, railways_edges = offsetIds(railways_nodes, railways_edges,
                                                   offset_node_id=nodes['id'].max() + 1,
                                                   offset_edge_id=edges['id'].max() + 1)
        nodes = pd.concat([nodes, railways_nodes], ignore_index=False,
                          verify_integrity=True, sort=False)
        edges = pd.concat([edges, railways_edges], ignore_index=False,
                          verify_integrity=True, sort=False)

    if "waterways" in transport_modes:
        logging.debug('Loading waterways data')
        waterways_nodes, waterways_edges = loadTransportData(filepaths, transport_params, "waterways")
        logging.debug(str(waterways_nodes.shape[0]) + " waterways nodes and " +
                      str(waterways_edges.shape[0]) + " waterways edges")
        waterways_nodes, waterways_edges = offsetIds(waterways_nodes, waterways_edges,
                                                     offset_node_id=nodes['id'].max() + 1,
                                                     offset_edge_id=edges['id'].max() + 1)
        nodes = pd.concat([nodes, waterways_nodes], ignore_index=False,
                          verify_integrity=True, sort=False)
        edges = pd.concat([edges, waterways_edges], ignore_index=False,
                          verify_integrity=True, sort=False)

    if "maritime" in transport_modes:
        logging.debug('Loading maritime data')
        maritime_nodes, maritime_edges = loadTransportData(filepaths, transport_params, "maritime")
        logging.debug(str(maritime_nodes.shape[0]) + " maritime nodes and " +
                      str(maritime_edges.shape[0]) + " maritime edges")
        maritime_nodes, maritime_edges = offsetIds(maritime_nodes, maritime_edges,
                                                   offset_node_id=nodes['id'].max() + 1,
                                                   offset_edge_id=edges['id'].max() + 1)
        nodes = pd.concat([nodes, maritime_nodes], ignore_index=False,
                          verify_integrity=True, sort=False)
        edges = pd.concat([edges, maritime_edges], ignore_index=False,
                          verify_integrity=True, sort=False)

    if "airways" in transport_modes:
        logging.debug('Loading airways data')
        airways_nodes, airways_edges = loadTransportData(filepaths, transport_params, "airways")
        logging.debug(str(airways_nodes.shape[0]) + " airways nodes and " +
                      str(airways_edges.shape[0]) + " airways edges")
        airways_nodes, airways_edges = offsetIds(airways_nodes, airways_edges,
                                                 offset_node_id=nodes['id'].max() + 1,
                                                 offset_edge_id=edges['id'].max() + 1)
        nodes = pd.concat([nodes, airways_nodes], ignore_index=False,
                          verify_integrity=True, sort=False)
        edges = pd.concat([edges, airways_edges], ignore_index=False,
                          verify_integrity=True, sort=False)

    if len(transport_modes) >= 2:
        logging.debug('Loading multimodal data')
        multimodal_edges = loadTransportData(filepaths, transport_params, "multimodal")
        multimodal_edges = filterMultimodalEdges(multimodal_edges, transport_modes)
        logging.debug(str(multimodal_edges.shape[0]) + " multimodal edges")
        multimodal_edges = assignEndpoints(multimodal_edges, nodes)
        multimodal_edges['id'] = multimodal_edges['id'] + edges['id'].max() + 1
        multimodal_edges.index = multimodal_edges['id']
        multimodal_edges.index.name = "index"
        edges = pd.concat([edges, multimodal_edges], ignore_index=False,
                          verify_integrity=True, sort=False)
        logging.debug('Total nb of transport nodes: ' + str(nodes.shape[0]))
        logging.debug('Total nb of transport edges: ' + str(edges.shape[0]))

    # Check conformity
    if (nodes['id'].duplicated().sum() > 0):
        raise ValueError('The following node ids are duplicated: ' +
                         nodes.loc[nodes['id'].duplicated(), "id"])
    if (edges['id'].duplicated().sum() > 0):
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
        T.add_transport_edge_with_nodes(road, edges, nodes)

    return T, nodes, edges
