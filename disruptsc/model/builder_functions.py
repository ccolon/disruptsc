from typing import TYPE_CHECKING

import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd
import geopandas as gpd
from pandas import Series

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.agents.country import Countries


def filter_sector(sector_table, cutoff_sector_output, cutoff_sector_demand,
                  combine_sector_cutoff, sectors_to_include, sectors_to_exclude, monetary_units_in_data):
    """Filter the sector table to sector whose output and/or final demand is larger than cutoff values
    In addition to filters, we can force to exclude or include some sectors

    Parameters
    ----------
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
    filtered_sectors_output = apply_sector_filter(sector_table, 'output', cutoff_sector_output,
                                                  units_in_data=monetary_units_in_data)
    filtered_sectors_demand = apply_sector_filter(sector_table, 'final_demand', cutoff_sector_demand,
                                                  units_in_data=monetary_units_in_data)

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
    if sectors_to_exclude:
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
    The way to cut_off is defined in cut_off_dic

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
                           transport_nodes: geopandas.GeoDataFrame, filepath_region_table: Path) -> pd.Series:
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


def get_long_lat(nodes_ids: pd.Series, transport_nodes: geopandas.GeoDataFrame) -> dict[str, Series]:
    od_point_table = transport_nodes[transport_nodes['id'].isin(nodes_ids)].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    return {
        'long': nodes_ids.map(road_node_id_to_long_lat['long']),
        'lat': nodes_ids.map(road_node_id_to_long_lat['long'])
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


def extract_final_list_of_sector(firms: "Firms"):
    n = len(firms)
    present_sectors = list(set([firm.sector for firm in firms.values()]))
    present_sectors.sort()
    flow_types_to_export = present_sectors + ['domestic_B2C', 'domestic_B2B', 'transit', 'import',
                                              'import_B2C', 'export', 'total']
    logging.info('Firm_list created, size is: ' + str(n))
    logging.info('Sectors present are: ' + str(present_sectors))
    return n, present_sectors, flow_types_to_export


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
