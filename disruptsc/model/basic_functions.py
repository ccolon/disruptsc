import logging
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree


def generate_weights(nb_suppliers: int, importance_of_each: list or None):
    # if there is only one supplier, return 1
    if nb_suppliers == 1:
        return [1]

    # if there are several and importance are provided, choose according to importance
    if importance_of_each is None:
        rdm_values = np.random.uniform(0, 1, size=nb_suppliers)
        return list(rdm_values / sum(rdm_values))

    # otherwise choose random values
    else:
        return [x / sum(importance_of_each) for x in importance_of_each]


def select_ids_and_weight(potential_supplier_ids: list, prob_to_be_selected: list, nb_suppliers_to_choose: int):
    """
    Selects a specified number of suppliers from a potential list based on their probabilities,
    and generates weights for the selected suppliers.

    Parameters:
        potential_supplier_ids (list): List of supplier IDs.
        prob_to_be_selected (list): Probability of each supplier being selected.
        nb_suppliers_to_choose (int): Number of suppliers to select.

    Returns:
        tuple: A tuple containing:
               - List of selected supplier IDs.
               - List of weights for the selected suppliers.
    """
    # Select supplier IDs based on their probabilities without replacement
    selected_ids = np.random.choice(potential_supplier_ids,
                                    p=prob_to_be_selected,
                                    size=nb_suppliers_to_choose,
                                    replace=False).tolist()
    # Map each supplier ID to its position in the original list
    index_map = {supplier_id: position for position, supplier_id in enumerate(potential_supplier_ids)}
    # Get the positions of the selected supplier IDs
    selected_positions = [index_map[supplier_id] for supplier_id in selected_ids]
    # Get the selection probabilities for the selected positions
    selected_prob = [prob_to_be_selected[position] for position in selected_positions]
    # Generate weights for the selected suppliers based on the selected probabilities
    weights = generate_weights(nb_suppliers_to_choose, selected_prob)

    return selected_ids, weights


def generate_weights_from_list(list_nb):
    sum_list = sum(list_nb)
    return [nb / sum_list for nb in list_nb]


def rescale_values(input_list, minimum=0.1, maximum=1, max_val=None, alpha=1, normalize=False):
    max_val = max_val or max(input_list)
    min_val = min(input_list)
    if max_val == min_val:
        res = [0.5 * maximum] * len(input_list)
    else:
        res = [
            minimum + (((val - min_val) / (max_val - min_val)) ** alpha) * (maximum - minimum)
            for val in input_list
        ]
    if normalize:
        res = [x / sum(res) for x in res]
    return res


def calculate_distance_between_agents(agentA, agentB):
    if (agentA.od_point == -1) or (agentB.od_point == -1):
        logging.warning("Try to calculate distance between agents, but one of them does not have real od point")
        return 1
    else:
        return compute_distance_from_arcmin(agentA.long, agentA.lat, agentB.long, agentB.lat)


def compute_distance_from_arcmin(x0, y0, x1, y1):
    # This is a very approximate way to convert arc distance into km
    EW_dist = (x1 - x0) * 112.5
    NS_dist = (y1 - y0) * 111
    return math.sqrt(EW_dist ** 2 + NS_dist ** 2)


def add_or_increment_dict_key(dic: dict, key, value: float | int):
    if key not in dic.keys():
        dic[key] = value
    else:
        dic[key] += value


def rescale_monetary_values(
        values: pd.Series | pd.DataFrame | float,
        time_resolution: str = "week",
        target_units: str = "mUSD",
        input_units: str = "USD"
) -> pd.Series | pd.DataFrame | float:
    """Rescale monetary values using the appropriate timescale and monetary units

    Parameters
    ----------
    values : pandas.Series, pandas.DataFrame, float
        Values to transform

    time_resolution : 'day', 'week', 'month', 'year'
        The number in the input table are yearly figure

    target_units : 'USD', 'kUSD', 'mUSD'
        Monetary units to which values are converted

    input_units : 'USD', 'kUSD', 'mUSD'
        Monetary units of the inputted values

    Returns
    -------
    same type as values
    """
    # Rescale according to the time period chosen
    periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
    values = values / periods[time_resolution]

    # Change units
    units = {"USD": 1, "kUSD": 1e3, "mUSD": 1e6}
    values = values * units[input_units] / units[target_units]

    return values


def add_or_append_to_dict(dictionary, key, value_to_add):
    if key in dictionary.keys():
        dictionary[key] += value_to_add
    else:
        dictionary[key] = value_to_add


def find_nearest_node_id(transport_nodes, gdf: gpd.GeoDataFrame, node_type='any'):
    """
    Finds the nearest road node for each point in final_gdf using KDTree.
    """
    if node_type == "road":
        transport_nodes = transport_nodes[transport_nodes['type'] == "road"].copy()
    transport_node_coords = np.array(list(transport_nodes.geometry.apply(lambda geom: (geom.x, geom.y))))
    gdf_coords = np.array(list(gdf.geometry.apply(lambda geom: (geom.x, geom.y))))

    kdtree = cKDTree(transport_node_coords)
    distances, indices = kdtree.query(gdf_coords)

    return transport_nodes.iloc[indices.tolist()].index.values


def mean_squared_distance(d1, d2):
    # Use the common keys in both dictionaries
    common_keys = set(d1.keys()) & set(d2.keys())
    if not common_keys:
        raise ValueError("No common keys between the dictionaries.")

    squared_diffs = [(d1[k] - d2[k]) ** 2 for k in common_keys]
    return math.sqrt(sum(squared_diffs) / len(squared_diffs)) / 1000
