import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import Series
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.agents.country import Countries


def rescale_monetary_values(
        values: pd.Series | pd.DataFrame | float,
        input_units: str = "USD",
        input_time_resolution: str = "year",
        target_units: str = "USD",
        target_time_resolution: str = "year"
) -> pd.Series | pd.DataFrame | float:
    """Rescale monetary values using the appropriate timescale and monetary units

    Parameters
    ----------
    target_time_resolution
    values : pandas.Series, pandas.DataFrame, float
        Values to transform

    input_time_resolution : 'day', 'week', 'month', 'year'
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

    values = values * periods[input_time_resolution] / periods[target_time_resolution]

    # Change units
    units = {"USD": 1, "kUSD": 1e3, "mUSD": 1e6}
    values = values * units[input_units] / units[target_units]

    return values


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
    """
    Rescale values to a specified range using NumPy vectorization for performance.
    
    Parameters
    ----------
    input_list : list or np.ndarray
        Input values to rescale
    minimum : float
        Minimum value of output range
    maximum : float  
        Maximum value of output range
    max_val : float, optional
        Override maximum value for scaling
    alpha : float
        Power scaling factor
    normalize : bool
        Whether to normalize result to sum to 1
        
    Returns
    -------
    list
        Rescaled values as list (for backward compatibility)
    """
    import numpy as np
    
    # Convert to numpy array for vectorized operations
    values = np.asarray(input_list, dtype=float)
    
    max_val = max_val if max_val is not None else np.max(values)
    min_val = np.min(values)
    
    if max_val == min_val:
        # All values are the same
        res = np.full_like(values, 0.5 * maximum)
    else:
        # Vectorized rescaling computation
        normalized = (values - min_val) / (max_val - min_val)
        res = minimum + (normalized ** alpha) * (maximum - minimum)
    
    if normalize:
        res = res / np.sum(res)
    
    return res.tolist()  # Return as list for backward compatibility


def calculate_distance_between_agents(agentA, agentB, transport_network=None):
    """
    Calculate distance between two agents.
    
    Parameters
    ----------
    agentA : BaseAgent
        First agent
    agentB : BaseAgent 
        Second agent
    transport_network : TransportNetwork, optional
        Transport network for cached od_point distances
        
    Returns
    -------
    float
        Distance in kilometers
    """
    if (agentA.od_point == -1) or (agentB.od_point == -1):
        logging.warning("Try to calculate distance between agents, but one of them does not have real od point")
        return 1
    
    # Use cached od_point distances if transport_network provided
    if transport_network is not None:
        return transport_network.get_distance_between_nodes(agentA.od_point, agentB.od_point)
    else:
        # Fallback to direct coordinate calculation
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


def add_or_append_to_dict(dictionary, key, value_to_add):
    if key in dictionary.keys():
        dictionary[key] += value_to_add
    else:
        dictionary[key] = value_to_add


def find_nearest_node_id(transport_nodes, gdf: gpd.GeoDataFrame):
    """
    Finds the nearest node for each point in final_gdf using KDTree.
    """
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


def draw_lognormal_samples(mean, coefficient_of_variation, N):
    """
    Generates N random samples from a lognormal distribution with given mean and coefficient of variation.

    Parameters:
    - mean (float): Mean of the distribution.
    - coefficient_of_variation (float): Coefficient of variation (CV = std_dev / mean).
    - N (int): Number of samples to draw.

    Returns:
    - samples (numpy array): Randomly drawn numbers following the specified lognormal distribution.
    """

    # Compute standard deviation
    std_dev = coefficient_of_variation * mean

    # Convert mean and std_dev to lognormal parameters
    mu = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std_dev**2 / mean**2)))

    # Draw samples from the lognormal distribution
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=N)

    return samples.tolist()


def find_min_in_nested_dict(d):
    """Recursively finds the minimum float value in a nested dictionary."""
    min_value = float("inf")  # Start with a very large value

    for key, value in d.items():
        if isinstance(value, dict):  # If value is a nested dictionary, recurse
            min_value = min(min_value, find_min_in_nested_dict(value))
        else:  # Leaf value (assumed to be float)
            min_value = min(min_value, value)

    return min_value


def load_sector_table(filepath: str) -> pd.DataFrame:
    """
    Load sector table from CSV file and ensure region_sector column exists.
    
    Parameters
    ----------
    filepath : str
        Path to the sector table CSV file
        
    Returns
    -------
    pd.DataFrame
        Sector table with region_sector column guaranteed to exist
    """
    sector_table = pd.read_csv(filepath)
    
    if "region_sector" not in sector_table.columns:
        sector_table['region_sector'] = sector_table['region'] + '_' + sector_table['sector']
    
    return sector_table


# Functions moved from builder_functions.py

def filter_sector(mrio, cutoff_sector_output, cutoff_sector_demand,
                  combine_sector_cutoff, sectors_to_include, sectors_to_exclude, monetary_units_in_data):
    """Filter the sector table to sector whose output and/or final demand is larger than cutoff values
    In addition to filters, we can force to exclude or include some sectors

    Parameters
    ----------
    mrio
    monetary_units_in_data
    sector_table : pandas.DataFrame
        Sector table
    cutoff_sector_output : dictionary
        Cutoff parameters for selecting the sectors based on output
        If type="percentage", the sector's output divided by all sectors' output is used
        If type="absolute", the sector's absolute output, in USD, is used
        If type="relative_to_average", the cutoff used is (cutoff value) * (country's total output) / (nb sectors)
    cutoff_sector_demand : dictionary
        Cutoff value for selecting the sectors based on final demand
        If type="percentage", the sector's final demand divided by all sectors' output is used
        If type="absolute", the sector's absolute output, in USD, is used
    combine_sector_cutoff: "and", "or"
        If 'and', select sectors that pass both the output and demand cutoff
        If 'or', select sectors that pass either the output or demand cutoff
    sectors_to_include : list of string or 'all'
        list of the sectors preselected by the user. Default to "all"
    sectors_to_exclude : list of string or None
        list of the sectors pre-eliminated by the user. Default to None

    Returns
    -------
    list of filtered sectors
    """
    # Select sectors based on output
    filtered_sectors_output = mrio.filter_industries_by_output(cutoff_sector_output['value'], cutoff_sector_output['type'],
                                                               cutoff_sector_output['unit'], monetary_units_in_data)
    filtered_sectors_demand = mrio.filter_industries_by_final_demand(cutoff_sector_demand['value'], cutoff_sector_demand['type'],
                                                               cutoff_sector_demand['unit'], monetary_units_in_data)

    # Merge both list
    if combine_sector_cutoff == 'and':
        filtered_sectors = list(set(filtered_sectors_output) & set(filtered_sectors_demand))
    elif combine_sector_cutoff == 'or':
        filtered_sectors = list(set(filtered_sectors_output + filtered_sectors_demand))
    else:
        raise ValueError("'combine_sector_cutoff' should be 'and' or 'or'")

    # Force to include some sector
    if isinstance(sectors_to_include, list):
        if len(set(sectors_to_include) - set(filtered_sectors)) > 0:
            selected_but_filtered_out_sectors = list(set(sectors_to_include) - set(filtered_sectors))
            logging.info("The following sectors were specifically selected but were filtered out" +
                         str(selected_but_filtered_out_sectors))
        filtered_sectors = list(set(sectors_to_include) & set(filtered_sectors))

    # Force to exclude some sectors
    if isinstance(sectors_to_exclude, list):
        filtered_sectors = [sector for sector in filtered_sectors if sector not in sectors_to_exclude]

    if len(filtered_sectors) == 0:
        raise ValueError("We excluded all sectors")

    # Sort list
    filtered_sectors.sort()
    return filtered_sectors


def get_absolute_cutoff_value(cutoff_dict: dict, units_in_data: str):
    assert cutoff_dict['type'] == "absolute"
    units = {"USD": 1, "kUSD": 1e3, "mUSD": 1e6}
    unit_adjusted_cutoff = cutoff_dict['value'] * units[cutoff_dict['unit']] / units[units_in_data]
    return unit_adjusted_cutoff


def apply_sector_filter(sector_table, filter_column, cut_off_dic, units_in_data):
    """Filter the sector_table using the filter_column
    The way to cut off is defined in cut_off_dic

    sector_table : pandas.DataFrame
        Sector table
    filter_column : string
        'output' or 'final_demand'
    cut_off_dic : dictionary
        Cutoff parameters for selecting the sectors based on output
        If type="percentage", the sector's filter_column divided by all sectors' output is used
        If type="absolute", the sector's absolute filter_column is used
        If type="relative_to_average", the cutoff used is (cutoff value) * (total filter_column) / (nb sectors)
    """
    sector_table_no_import = sector_table[sector_table['sector'] != "IMP"]

    if cut_off_dic['type'] == "percentage":
        rel_output = sector_table_no_import[filter_column] / sector_table_no_import['output'].sum()
        filtered_sectors = sector_table_no_import.loc[
            rel_output > cut_off_dic['value'],
            "sector"
        ].tolist()
    elif cut_off_dic['type'] == "absolute":
        unit_adjusted_cutoff = get_absolute_cutoff_value(cut_off_dic, units_in_data)
        filtered_sectors = sector_table_no_import.loc[
            sector_table_no_import[filter_column] > unit_adjusted_cutoff,
            "sector"
        ].tolist()
    elif cut_off_dic['type'] == "relative_to_average":
        cutoff = cut_off_dic['value'] \
                 * sector_table_no_import[filter_column].sum() \
                 / sector_table_no_import.shape[0]
        filtered_sectors = sector_table_no_import.loc[
            sector_table_no_import['output'] > cutoff,
            "sector"
        ].tolist()
    else:
        raise ValueError("cutoff type should be 'percentage', 'absolute', or 'relative_to_average'")
    if len(filtered_sectors) == 0:
        raise ValueError("The output cutoff value is so high that it filtered out all sectors")
    return filtered_sectors


def get_closest_road_nodes(regions: pd.Series,
                           transport_nodes: gpd.GeoDataFrame, filepath_region_table: Path) -> pd.Series:
    region_table = gpd.read_file(filepath_region_table)
    dic_region_to_points = region_table.set_index('region')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_region_to_points.items()
    }
    closest_road_nodes = regions.map(dic_region_to_road_node_id)
    if closest_road_nodes.isnull().sum() > 0:
        raise KeyError(f"{closest_road_nodes.isnull().sum()} regions not found: "
                       f"{regions[closest_road_nodes.isnull()].to_list()}")
    return closest_road_nodes


def get_long_lat(nodes_ids: pd.Series, transport_nodes: gpd.GeoDataFrame) -> dict[str, Series]:
    od_point_table = transport_nodes[transport_nodes['id'].isin(nodes_ids)].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    return {
        'long': nodes_ids.map(road_node_id_to_long_lat['long']),
        'lat': nodes_ids.map(road_node_id_to_long_lat['lat'])
    }


def get_index_closest_point(point, df_with_points):
    """Given a point it finds the index of the closest points in a Point GeoDataFrame.

    Parameters
    ----------
    point: shapely.Point
        Point object of which we want to find the closest point
    df_with_points: geopandas.GeoDataFrame
        containing the points among which we want to find the
        one that is the closest to point

    Returns
    -------
    type depends on the index data type of df_with_points
        index object of the closest point in df_with_points
    """
    distance_list = [point.distance(item) for item in df_with_points['geometry'].tolist()]
    return df_with_points.index[distance_list.index(min(distance_list))]


def load_ton_usd_equivalence(sector_table: pd.DataFrame, firm_table: pd.DataFrame,
                             firms: "Firms", countries: "Countries"):
    """Load equivalence between usd and ton

    It updates the firms and countries.
    It updates the 'usd_per_ton' attribute of firms, based on their sector.
    It updates the 'usd_per_ton' attribute of countries, it gives the average.
    Note that this will be applied only to goods that are delivered by those agents.

    sector_table : pandas.DataFrame
        Sector table
    firms : list(Firm objects)
        list of firms
    countries : list(Country objects)
        list of countries
    """
    sector_to_usd_per_ton = sector_table.set_index('sector')['usd_per_ton']
    firm_table['usd_per_ton'] = firm_table['region_sector'].map(sector_to_usd_per_ton)
    for firm in firms.values():
        firm.usd_per_ton = sector_to_usd_per_ton[firm.region_sector]

    for country in countries.values():
        country.usd_per_ton = sector_to_usd_per_ton['IMP']
