import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd
import geopandas as gpd
import numpy as np
from pandas import Series

from code.agents.firm import Firm, FirmList
from code.agents.household import Household, HouseholdList
from code.agents.country import Country, CountryList


def filter_sector(sector_table, cutoff_sector_output, cutoff_sector_demand,
                  combine_sector_cutoff='and', sectors_to_include="all", sectors_to_exclude=None):
    """Filter the sector table to sector whose output and/or final demand is larger than cutoff values
    In addition to filters, we can force to exclude or include some sectors

    Parameters
    ----------
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
    filtered_sectors_output = apply_sector_filter(sector_table, 'output', cutoff_sector_output)
    filtered_sectors_demand = apply_sector_filter(sector_table, 'final_demand', cutoff_sector_demand)

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


def apply_sector_filter(sector_table, filter_column, cut_off_dic):
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
        filtered_sectors = sector_table_no_import.loc[
            sector_table_no_import[filter_column] > cut_off_dic['value'],
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


def get_closest_road_nodes(admin_unit_ids: pd.Series,
                           transport_nodes: geopandas.GeoDataFrame, filepath_region_table: Path) -> pd.Series:
    region_table = gpd.read_file(filepath_region_table)
    dic_region_to_points = region_table.set_index('admin_code')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_region_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_region_to_points.items()
    }
    closest_road_nodes = admin_unit_ids.map(dic_region_to_road_node_id)
    if closest_road_nodes.isnull().sum() > 0:
        logging.warning(f"{closest_road_nodes.isnull().sum()} admin_units not found")
        raise KeyError(f"{closest_road_nodes.isnull().sum()} admin_units not found: "
                       f"{admin_unit_ids[closest_road_nodes.isnull()].to_list()}")
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
        GeoDataFrame containing the points among which we want to find the
        one that is the closest to point

    Returns
    -------
    type depends on the index data type of df_with_points
        index object of the closest point in df_with_points
    """
    distance_list = [point.distance(item) for item in df_with_points['geometry'].tolist()]
    return df_with_points.index[distance_list.index(min(distance_list))]


def extract_final_list_of_sector(firm_list: FirmList):
    n = len(firm_list)
    present_sectors = list(set([firm.main_sector for firm in firm_list]))
    present_sectors.sort()
    flow_types_to_export = present_sectors + ['domestic_B2C', 'domestic_B2B', 'transit', 'import', 'export', 'total']
    logging.info('Firm_list created, size is: ' + str(n))
    logging.info('Sectors present are: ' + str(present_sectors))
    return n, present_sectors, flow_types_to_export


def define_households_from_mrio_data(
        sector_table: pd.DataFrame,
        filepath_region_table: Path,
        filtered_sectors: list,
        local_demand_cutoff: float,
        transport_nodes: gpd.GeoDataFrame,
        time_resolution: str,
        target_units: str,
        input_units: str
):
    # household_table = gpd.read_file(filepath_region_table)
    # household_table = location_table.copy(deep=False)
    # TODO ajouter la demande finale par secteur country_sector_table
    # print(household_table)
    final_demand = sector_table["final_demand"]
    # household_table = household_table.stack() #.transpose()

    # TEST

    household_table = gpd.read_file(filepath_region_table)

    # Add final demand
    final_demand = sector_table.loc[sector_table['sector'].isin(filtered_sectors), ['sector', 'final_demand']]
    final_demand_as_row = final_demand.set_index('sector').transpose()

    # Duplicate rows
    household_table = pd.concat([household_table] * len(final_demand_as_row))

    # Align index and concat
    household_table.index = final_demand_as_row.index
    household_table = pd.concat([household_table, final_demand_as_row], axis=1)

    # Reshape the household table
    household_table = household_table.stack().reset_index()
    household_table.columns = ['admin_code', 'column', 'value']

    print(household_table)

    ## Fin TEST

    # C. Create one household per OD point
    logging.info('Assigning households to od-points')
    dic_select_admin_unit_to_points = household_table.set_index('admin_code')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic
    dic_admin_unit_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_select_admin_unit_to_points.items()
    }
    # Map household to closest road nodes
    household_table['od_point'] = household_table['admin_code'].map(dic_admin_unit_to_road_node_id)

    # Combine households that are in the same od-point
    household_table = household_table \
        .drop(columns=['geometry', 'admin_code']) \
        .groupby('od_point', as_index=False) \
        .sum()
    logging.info(str(household_table.shape[0]) + ' od-point selected for demand')

    # D. Filter out small demand
    if local_demand_cutoff > 0:
        household_table[filtered_sectors] = household_table[filtered_sectors].mask(
            household_table[filtered_sectors] < local_demand_cutoff
        )
    # info
    logging.info('Create ' + str(household_table.shape[0]) + " households in " +
                 str(household_table['od_point'].nunique()) + ' od points')
    for sector in filtered_sectors:
        logging.info('Sector ' + sector + ": create " +
                     str((~household_table[sector].isnull()).sum()) +
                     " buying households that covers " +
                     "{:.0f}%".format(
                         household_table[sector].sum() \
                         / sector_table.set_index('sector').loc[sector, 'final_demand'] * 100
                     ) + " of total final demand"
                     )
    if (household_table[filtered_sectors].sum(axis=1) == 0).any():
        logging.warning('Some households have no purchase plan because of all their sectoral demand below cutoff!')

    # E. Add information required by the createHouseholds function
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(household_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_longlat = od_point_table.set_index('id')[['long', 'lat']]
    household_table['long'] = household_table['od_point'].map(road_node_id_to_longlat['long'])
    household_table['lat'] = household_table['od_point'].map(road_node_id_to_longlat['lat'])
    # add id
    household_table['id'] = list(range(household_table.shape[0]))

    # F. Create purchase plan per household
    # rescale according to time resolution
    household_table[filtered_sectors] = rescale_monetary_values(
        household_table[filtered_sectors],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # to dict
    household_sector_consumption = household_table.set_index('id')[filtered_sectors].to_dict(orient='index')
    # remove nan values
    household_sector_consumption = {
        i: {
            sector: amount
            for sector, amount in purchase_plan.items()
            if ~np.isnan(amount)
        }
        for i, purchase_plan in household_sector_consumption.items()
    }

    return household_table, household_sector_consumption


def define_households_from_mrio(
        filepath_mrio: Path,
        filepath_region_table: Path,
        transport_nodes: gpd.GeoDataFrame,
        time_resolution: str,
        target_units: str,
        input_units: str
):
    # Load mrio
    mrio = pd.read_csv(filepath_mrio, index_col=0)
    # Extract region_households
    region_households = [col for col in mrio.columns if
                         col[-2:] == "-H"]  # format -H... :TODO a bit specific to Ecuador, change

    # Create household table
    household_table = pd.DataFrame({"name": region_households})
    household_table['region'] = household_table['name'].str.extract('([0-9]*)-H')
    logging.info(f"Select {household_table.shape[0]} firms in {household_table['region'].nunique()} admin units")

    # Identify OD point
    household_table['od_point'] = get_closest_road_nodes(household_table['region'], transport_nodes,
                                                         filepath_region_table)

    # Add long lat
    long_lat = get_long_lat(household_table['od_point'], transport_nodes)
    household_table['long'] = long_lat['long']
    household_table['lat'] = long_lat['lat']

    # Add id
    household_table['id'] = list(range(household_table.shape[0]))

    # Identify final demand per region_sector
    mrio = rescale_monetary_values(
        mrio,
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    col_final_demand = [col for col in mrio.columns if col[-2:] == "-H"]
    household_sector_consumption = mrio[col_final_demand].stack().reset_index() \
        .groupby('level_1') \
        .apply(lambda df: df.set_index('level_0')[0].to_dict()) \
        .to_dict()
    # Replace name by id :TODO use name as id to makes this unecessary
    dic_name_to_id = household_table.set_index('name')['id']
    household_sector_consumption = {
        dic_name_to_id[name]: value
        for name, value in household_sector_consumption.items()
    }

    # Info
    logging.info(f"Create {household_table.shape[0]} households in {household_table['od_point'].nunique()} od points")

    return household_table, household_sector_consumption


def define_households(
        sector_table: pd.DataFrame,
        filepath_admin_unit_data: Path,
        filtered_sectors: list,
        pop_cutoff: float,
        pop_density_cutoff: float,
        local_demand_cutoff: float,
        transport_nodes: gpd.GeoDataFrame,
        time_resolution: str,
        target_units: str,
        input_units: str
):
    '''Define the number of households to model and their purchase plan based on input demographic data
    and filtering options

    Parameters
    ---------
    input_units
    target_units
    time_resolution
    transport_nodes
    local_demand_cutoff
    pop_density_cutoff
    pop_cutoff
    filtered_sectors
    filepath_admin_unit_data
    sector_table

    Returns
    -------
    household_table
    household_purchase_plan

    '''
    # A. Filter admin unit based on density
    # load file
    admin_unit_data = gpd.read_file(filepath_admin_unit_data)
    # filter & keep household were firms are
    cond = pd.Series(True, index=admin_unit_data.index)
    if pop_density_cutoff > 0:
        cond = cond & (admin_unit_data['pop_density'] >= pop_density_cutoff)
    if pop_cutoff > 0:
        cond = cond & (admin_unit_data['population'] >= pop_cutoff)
    # create household_table
    household_table = admin_unit_data.loc[cond, ['population', 'geometry', 'admin_code']].copy()
    logging.info(
        str(cond.sum()) + ' admin units selected over ' + str(admin_unit_data.shape[0]) + ' representing ' +
        "{:.0f}%".format(household_table['population'].sum() / admin_unit_data['population'].sum() * 100) +
        ' of population'
    )

    # B. Add final demand
    # get final demand for the selected sector
    final_demand = sector_table.loc[sector_table['sector'].isin(filtered_sectors), ['sector', 'final_demand']]
    # put as single row
    final_demand_as_row = final_demand.set_index('sector').transpose()
    # duplicates rows
    final_demand_each_household = pd.concat([final_demand_as_row for i in range(household_table.shape[0])])
    # align index and concat
    final_demand_each_household.index = household_table.index
    # compute final demand per admin unit
    rel_pop = household_table['population'] / admin_unit_data['population'].sum()
    final_demand_each_household = final_demand_each_household.multiply(rel_pop, axis='index')
    # add to household table
    household_table = pd.concat([household_table, final_demand_each_household], axis=1)

    # C. Create one household per OD point
    logging.info('Assigning households to od-points')
    dic_select_admin_unit_to_points = household_table.set_index('admin_code')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic
    dic_admin_unit_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_select_admin_unit_to_points.items()
    }
    # Map household to closest road nodes
    household_table['od_point'] = household_table['admin_code'].map(dic_admin_unit_to_road_node_id)
    # Combine households that are in the same od-point
    household_table = household_table \
        .drop(columns=['geometry', 'admin_code']) \
        .groupby('od_point', as_index=False) \
        .sum()
    logging.info(str(household_table.shape[0]) + ' od-point selected for demand')

    # D. Filter out small demand
    if local_demand_cutoff > 0:
        household_table[filtered_sectors] = household_table[filtered_sectors].mask(
            household_table[filtered_sectors] < local_demand_cutoff
        )
    # info
    logging.info('Create ' + str(household_table.shape[0]) + " households in " +
                 str(household_table['od_point'].nunique()) + ' od points')
    for sector in filtered_sectors:
        logging.info('Sector ' + sector + ": create " +
                     str((~household_table[sector].isnull()).sum()) +
                     " buying households that covers " +
                     "{:.0f}%".format(
                         household_table[sector].sum() \
                         / sector_table.set_index('sector').loc[sector, 'final_demand'] * 100
                     ) + " of total final demand"
                     )
    if (household_table[filtered_sectors].sum(axis=1) == 0).any():
        logging.warning('Some households have no purchase plan because of all their sectoral demand below cutoff!')

    # E. Add information required by the createHouseholds function
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(household_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_longlat = od_point_table.set_index('id')[['long', 'lat']]
    household_table['long'] = household_table['od_point'].map(road_node_id_to_longlat['long'])
    household_table['lat'] = household_table['od_point'].map(road_node_id_to_longlat['lat'])
    # add id
    household_table['id'] = list(range(household_table.shape[0]))
    # add name, not really useful
    household_table['name'] = household_table['od_point'].astype(str) + "-H"

    # F. Create purchase plan per household
    # rescale according to time resolution
    household_table[filtered_sectors] = rescale_monetary_values(
        household_table[filtered_sectors],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # to dict
    household_sector_consumption = household_table.set_index('id')[filtered_sectors].to_dict(orient='index')
    # remove nan values
    household_sector_consumption = {
        i: {
            sector: amount
            for sector, amount in purchase_plan.items()
            if ~np.isnan(amount)
        }
        for i, purchase_plan in household_sector_consumption.items()
    }

    return household_table, household_sector_consumption


def create_households(
        household_table: pd.DataFrame,
        household_sector_consumption: dict
):
    """Create the households

    It uses household_table & household_sector_consumption from defineHouseholds

    Parameters
    ----------
    household_table: pandas.DataFrame
        household_table
    household_sector_consumption: dic
        {<household_id>: {<sector>: <amount>}}

    Returns
    -------
    list of Household
    """

    logging.debug('Creating household_list')
    household_table = household_table.set_index('id')
    household_list = HouseholdList([
        Household('hh_' + str(i),
                  name=household_table.loc[i, "name"],
                  odpoint=household_table.loc[i, "od_point"],
                  long=float(household_table.loc[i, 'long']),
                  lat=float(household_table.loc[i, 'lat']),
                  sector_consumption=household_sector_consumption[i]
                  )
        for i in household_table.index.tolist()
    ])
    logging.info('Households generated')

    return household_list


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


def add_households_for_firms(
        firm_table: pd.DataFrame,
        household_table: pd.DataFrame,
        filepath_admin_unit_data: str,
        sector_table: pd.DataFrame,
        filtered_sectors: list,
        time_resolution: str,
        target_units: str,
        input_units: str
):
    """
    We suppose that all firms face the same final demand. i.e., all firms are both B2B and B2C
    We suppose that households only buy more firms located on the same point
    Some firms are located in areas where there are no households
    Therefore, they cannot sell their final demand
    So this function add households in those points
    """

    # A. Examine where we need new households
    cond_no_household = ~firm_table['od_point'].isin(household_table['od_point'])
    logging.info(cond_no_household.sum(), 'firms without local households')
    logging.info(firm_table.loc[cond_no_household, 'od_point'].nunique(), 'od points with firms without households')

    # B. Create new household table
    added_household_table = firm_table[cond_no_household].groupby('od_point', as_index=False)['population'].max()
    od_point_long_lat = firm_table.loc[cond_no_household, ["od_point", 'long', "lat"]].drop_duplicates()
    added_household_table = added_household_table.merge(od_point_long_lat, how='left', on='odpoint')

    # B1. Load admin data to get tot population
    admin_unit_data = gpd.read_file(filepath_admin_unit_data)
    tot_pop = admin_unit_data['population'].sum()

    # B2. Load sector table to get final demand
    final_demand = sector_table.loc[sector_table['sector'].isin(filtered_sectors), ['sector', 'final_demand']]

    # B3. Add final demand per sector per new household
    # put as single row
    final_demand_as_row = final_demand.set_index('sector').transpose()
    # duplicates rows
    final_demand_each_household = pd.concat([final_demand_as_row for i in range(added_household_table.shape[0])])
    # align index and concat
    final_demand_each_household.index = added_household_table.index
    # compute final demand per commune
    rel_pop = added_household_table['population'] / tot_pop
    final_demand_each_household = final_demand_each_household.multiply(rel_pop, axis='index')
    # keep demand only for firm that are there
    cond_to_reject = (
        firm_table[cond_no_household].groupby(['od_point', 'sector'])['population'].sum().isnull()).unstack('sector')
    cond_to_reject = cond_to_reject.fillna(True)
    cond_to_reject.index = final_demand_each_household.index
    final_demand_each_household = final_demand_each_household.mask(cond_to_reject)
    # add to household table
    added_household_table = pd.concat([added_household_table, final_demand_each_household], axis=1)
    # rescale according to time resolution
    added_household_table[filtered_sectors] = rescale_monetary_values(
        added_household_table[filtered_sectors],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # C. Merge
    added_household_table['id'] = [household_table['id'].max() + 1 + i for i in range(added_household_table.shape[0])]
    added_household_table.index = added_household_table['id']
    household_table = pd.concat([household_table, added_household_table], sort=True)
    # household_table.to_csv('tt.csv')

    # D. Log info
    logging.info('Create ' + str(household_table.shape[0]) + " households in " +
                 str(household_table['odpoint'].nunique()) + ' od points')
    for sector in filtered_sectors:
        tot_demand = rescale_monetary_values(sector_table.set_index('sector').loc[sector, 'final_demand'],
                                             time_resolution=time_resolution, target_units=target_units,
                                             input_units=input_units)
        logging.info('Sector ' + sector + ": create " +
                     str((~household_table[sector].isnull()).sum()) +
                     " buying households that covers " +
                     "{:.0f}%".format(
                         household_table[sector].sum() \
                         / tot_demand * 100
                     ) + " of total final demand"
                     )
    if (household_table[filtered_sectors].sum(axis=1) == 0).any():
        logging.warning('Some households have no purchase plan!')

    # E. Create household_sector_consumption dic
    # create dic
    household_sector_consumption = household_table.set_index('id')[filtered_sectors].to_dict(orient='index')
    # remove nan values
    household_sector_consumption = {
        i: {
            sector: amount
            for sector, amount in purchase_plan.items()
            if ~np.isnan(amount)
        }
        for i, purchase_plan in household_sector_consumption.items()
    }

    return household_table, household_sector_consumption


def load_ton_usd_equivalence(sector_table: pd.DataFrame, firm_list: FirmList, country_list: CountryList):
    """Load equivalence between usd and ton

    It updates the firm_list and country_list.
    It updates the 'usd_per_ton' attribute of firms, based on their sector.
    It updates the 'usd_per_ton' attribute of countries, it gives the average.
    Note that this will be applied only to goods that are delivered by those agents.

    sector_table : pandas.DataFrame
        Sector table
    firm_list : list(Firm objects)
        list of firms
    country_list : list(Country objects)
        list of countries
    """
    sector_to_usd_per_ton = sector_table.set_index('sector')['usd_per_ton']
    for firm in firm_list:
        firm.usd_per_ton = sector_to_usd_per_ton[firm.main_sector]

    for country in country_list:
        country.usd_per_ton = sector_to_usd_per_ton.mean()

