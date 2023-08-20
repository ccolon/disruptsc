import warnings
import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd
import geopandas as gpd
import numpy as np

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


def define_firms_from_local_economic_data(filepath_admin_unit_economic_data: Path,
                                          sectors_to_include: list, transport_nodes: geopandas.GeoDataFrame,
                                          filepath_sector_table: Path, min_nb_firms_per_sector: int):
    """Define firms based on the admin_unit_economic_data.
    The output is a dataframe, 1 row = 1 firm.
    The instances Firms are created in the createFirm function.

    Steps:
    1. Load the admin_unit_economic_data
    2. It adds a row only when the sector in one admin_unit is higher than the sector_cutoffs
    3. It identifies the node of the road network that is the closest to the admin_unit point
    4. It combines the firms of the same sector that are in the same road node (case of 2 admin_unit close to the same road node)
    5. It calculates the "importance" of each firm = their size relative to the sector size

    Parameters
    ----------
    min_nb_firms_per_sector
    filepath_admin_unit_economic_data: string
        Path to the district_data table
    sectors_to_include: list or 'all'
        if 'all', include all sectors, otherwise define the list of sector to include
    transport_nodes: geopandas.GeoDataFrame
        transport nodes resulting from createTransportNetwork
    filepath_sector_table: string
        Path to the sector table
    """

    # A. Create firm table
    # A.1. load files
    admin_unit_eco_data = gpd.read_file(filepath_admin_unit_economic_data)
    sector_table = pd.read_csv(filepath_sector_table)

    # A.2. for each sector, select admin_unit where supply_data is over threshold
    # and populate firm table
    firm_table_per_admin_unit = pd.DataFrame()
    for sector, row in sector_table.set_index("sector").iterrows():
        if (sectors_to_include == "all") or (sector in sectors_to_include):
            # check that the supply metric is in the data
            if row["supply_data"] not in admin_unit_eco_data.columns:
                logging.warning(f"{row['supply_data']} for sector {sector} is missing from the economic data. "
                                f"We will create by default firms in the {min_nb_firms_per_sector} "
                                f"most populated admin units")
                where_create_firm = admin_unit_eco_data["population"].nlargest(min_nb_firms_per_sector).index
                # populate firm table
                new_firm_table = pd.DataFrame({
                    "sector": sector,
                    "admin_unit": admin_unit_eco_data.loc[where_create_firm, "admin_code"].tolist(),
                    "population": admin_unit_eco_data.loc[where_create_firm, "population"].tolist(),
                    "absolute_size": admin_unit_eco_data.loc[where_create_firm, "population"].tolist()
                })
            else:
                # create one firm where economic metric is over threshold
                where_create_firm = admin_unit_eco_data[row["supply_data"]] > row["cutoff"]
                # if it results in less than 5 firms, we go below the cutoff to get at least 5 firms,
                # only if there are enough admin_units with positive supply_data
                if where_create_firm.sum() < min_nb_firms_per_sector:
                    cond_positive_supply_data = admin_unit_eco_data[row["supply_data"]] > 0
                    where_create_firm = admin_unit_eco_data.loc[cond_positive_supply_data, row["supply_data"]].nlargest(
                        min_nb_firms_per_sector).index
                # populate firm table
                new_firm_table = pd.DataFrame({
                    "sector": sector,
                    "admin_unit": admin_unit_eco_data.loc[where_create_firm, "admin_code"].tolist(),
                    "population": admin_unit_eco_data.loc[where_create_firm, "population"].tolist(),
                    "absolute_size": admin_unit_eco_data.loc[where_create_firm, row["supply_data"]]
                })

            new_firm_table['relative_size'] = new_firm_table['absolute_size'] / new_firm_table['absolute_size'].sum()
            firm_table_per_admin_unit = pd.concat([firm_table_per_admin_unit, new_firm_table], axis=0)

    # B. Assign firms to the closest road nodes
    # B.1. Create a dictionary that link a admin_unit to id of the closest road node
    # Create dic that links admin_unit to points
    selected_admin_units = list(firm_table_per_admin_unit['admin_unit'].unique())
    logging.info('Select ' + str(firm_table_per_admin_unit.shape[0]) +
                 " in " + str(len(selected_admin_units)) + ' admin units')
    cond = admin_unit_eco_data['admin_code'].isin(selected_admin_units)
    logging.info('Assigning firms to od-points')
    dic_selected_admin_unit_to_points = admin_unit_eco_data[cond].set_index('admin_code')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic
    dic_admin_unit_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_selected_admin_unit_to_points.items()
    }

    # B.2. Map firm to the closest road node
    firm_table_per_admin_unit['od_point'] = firm_table_per_admin_unit['admin_unit'].map(dic_admin_unit_to_road_node_id)

    # C. Combine firms that are in the same od-point and in the same sector
    # group by od-point and sector
    firm_table_per_od_point = firm_table_per_admin_unit \
        .drop(columns='admin_unit') \
        .groupby(['od_point', 'sector'], as_index=False) \
        .sum()

    # D. Add information required by the createFirms function
    # add sector type
    sector_to_sector_type = sector_table.set_index('sector')['type']
    firm_table_per_od_point['sector_type'] = firm_table_per_od_point['sector'].map(sector_to_sector_type)
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table_per_od_point['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_longlat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table_per_od_point['long'] = firm_table_per_od_point['od_point'].map(road_node_id_to_longlat['long'])
    firm_table_per_od_point['lat'] = firm_table_per_od_point['od_point'].map(road_node_id_to_longlat['lat'])
    # add id
    firm_table_per_od_point['id'] = list(range(firm_table_per_od_point.shape[0]))
    # add importance
    firm_table_per_od_point['importance'] = firm_table_per_od_point['relative_size']

    # # E. Add final demand per firm
    # # evaluate share of population represented
    # cond = admin_unit_eco_data['admin_code'].isin(selected_admin_units)
    # represented_pop = admin_unit_eco_data.loc[cond, 'population'].sum()
    # total_population = admin_unit_eco_data['population'].sum()
    # # evaluate final demand
    # rel_pop = firm_table['population'] / total_population
    # tot_demand_of_sector = firm_table['sector'].map(sector_table.set_index('sector')['final_demand'])
    # firm_table['final_demand'] = rel_pop * tot_demand_of_sector
    # # print info
    # logging.info("{:.0f}%".format(represented_pop / total_population * 100)+
    #     " of population represented")
    # logging.info("{:.0f}%".format(firm_table['final_demand'].sum() / sector_table['final_demand'].sum() * 100)+
    #     " of final demand is captured")
    # logging.info("{:.0f}%".format(firm_table['final_demand'].sum() / \
    #     sector_table.set_index('sector').loc[sectors_to_include, 'final_demand'].sum() * 100)+
    #     " of final demand of selected sector is captured")

    # F. Log information
    logging.info('Create ' + str(firm_table_per_od_point.shape[0]) + " firms in " +
                 str(firm_table_per_od_point['od_point'].nunique()) + ' od points')
    for sector, row in sector_table.set_index("sector").iterrows():
        if (sectors_to_include == "all") or (sector in sectors_to_include):
            if row["supply_data"] in admin_unit_eco_data.columns:
                cond = firm_table_per_od_point['sector'] == sector
                logging.info(f"Sector {sector}: create {cond.sum()} firms that covers " +
                             "{:.0f}%".format(firm_table_per_od_point.loc[cond, 'absolute_size'].sum()
                                              / admin_unit_eco_data[row['supply_data']].sum() * 100) +
                             f" of total {row['supply_data']}")
            else:
                cond = firm_table_per_od_point['sector'] == sector
                logging.info(f"Sector {sector}: since {row['supply_data']} is not in the admin data, "
                             f"create {cond.sum()} firms that covers " +
                             "{:.0f}%".format(firm_table_per_od_point.loc[cond, 'population'].sum()
                                              / admin_unit_eco_data["population"].sum() * 100) +
                             f" of population")

    return firm_table_per_od_point, firm_table_per_admin_unit


def create_firms(
        firm_table: pd.DataFrame,
        keep_top_n_firms: object = None,
        reactivity_rate: float = 0.1,
        utilization_rate: float = 0.8
):
    """Create the firms

    It uses firm_table from rescaleNbFirms

    Parameters
    ----------
    firm_table: pandas.DataFrame
        firm_table from rescaleNbFirms
    keep_top_n_firms: None (default) or integer
        (optional) can be specified if we want to keep only the first n firms, for testing purposes
    reactivity_rate: float
        Determines the speed at which firms try to reach their inventory duration target. Default to 0.1.
    utilization_rate: float
        Set the utilization rate, which determines the production capacity at the input-output equilibrium.

    Returns
    -------
    list of Firms
    """

    if isinstance(keep_top_n_firms, int):
        firm_table = firm_table.iloc[:keep_top_n_firms, :]

    logging.debug('Creating firm_list')
    ids = firm_table['id'].tolist()
    firm_table = firm_table.set_index('id')
    firm_list = FirmList([
        Firm(i,
             sector=firm_table.loc[i, "sector"],
             sector_type=firm_table.loc[i, "sector_type"],
             odpoint=firm_table.loc[i, "od_point"],
             importance=firm_table.loc[i, 'importance'],
             # geometry=firm_table.loc[i, 'geometry'],
             long=float(firm_table.loc[i, 'long']),
             lat=float(firm_table.loc[i, 'lat']),
             utilization_rate=utilization_rate,
             reactivity_rate=reactivity_rate
             )
        for i in ids
    ])
    # We add a bit of noise to the long and lat coordinates
    # It allows to visually disentangle firms located at the same od-point when plotting the map.
    for firm in firm_list:
        firm.add_noise_to_geometry()

    return firm_list


def define_firms_from_mrio_data(
        filepath_country_sector_table: str,
        filepath_region_table: str,
        transport_nodes: pd.DataFrame):
    # Load firm table
    firm_table = pd.read_csv(filepath_country_sector_table)

    # TODO: duplicate lines
    # Duplicate the lines by concatenating the DataFrame with itself
    firm_table = pd.concat([firm_table] * 2, ignore_index=True)
    # Assign firms to closest road nodes
    selected_admin_units = list(firm_table['country_ISO'].unique())
    logging.info('Select ' + str(firm_table.shape[0]) +
                 " firms in " + str(len(selected_admin_units)) + ' admin units')
    location_table = gpd.read_file(filepath_region_table)
    cond_selected_admin_units = location_table['country_ISO'].isin(selected_admin_units)
    dic_admin_unit_to_points = location_table[cond_selected_admin_units].set_index('country_ISO')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_admin_unit_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_admin_unit_to_points.items()
    }
    firm_table['od_point'] = firm_table['country_ISO'].map(dic_admin_unit_to_road_node_id)

    # Information required by the createFirms function
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_longlat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table['long'] = firm_table['od_point'].map(road_node_id_to_longlat['long'])
    firm_table['lat'] = firm_table['od_point'].map(road_node_id_to_longlat['lat'])
    # add importance
    firm_table['importance'] = 10

    # add id (where is it created otherwise?)
    firm_table = firm_table.reset_index().rename(columns={"index": "id"})

    return firm_table


def define_firms_from_network_data(
        filepath_firm_table: str,
        filepath_location_table: str,
        sectors_to_include: list,
        transport_nodes: pd.DataFrame,
        filepath_sector_table: str
):
    '''Define firms based on the firm_table
    The output is a dataframe, 1 row = 1 firm.
    The instances Firms are created in the createFirm function.
    '''
    # Load firm table
    firm_table = pd.read_csv(filepath_firm_table, dtype={'adminunit': str})

    # Filter out some sectors
    if sectors_to_include != "all":
        firm_table = firm_table[firm_table['sector'].isin(sectors_to_include)]

    # Assign firms to closest road nodes
    selected_admin_units = list(firm_table['adminunit'].unique())
    logging.info('Select ' + str(firm_table.shape[0]) +
                 " firms in " + str(len(selected_admin_units)) + ' admin units')
    location_table = gpd.read_file(filepath_location_table)
    cond_selected_admin_units = location_table['admin_code'].isin(selected_admin_units)
    dic_admin_unit_to_points = location_table[cond_selected_admin_units].set_index('admin_code')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_admin_unit_to_road_node_id = {
        admin_unit: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for admin_unit, point in dic_admin_unit_to_points.items()
    }
    firm_table['odpoint'] = firm_table['adminunit'].map(dic_admin_unit_to_road_node_id)

    # Information required by the createFirms function
    # add sector type
    sector_table = pd.read_csv(filepath_sector_table)
    sector_to_sector_type = sector_table.set_index('sector')['type']
    firm_table['sector_type'] = firm_table['sector'].map(sector_to_sector_type)
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table['odpoint'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_longlat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table['long'] = firm_table['odpoint'].map(road_node_id_to_longlat['long'])
    firm_table['lat'] = firm_table['odpoint'].map(road_node_id_to_longlat['lat'])
    # add importance
    firm_table['importance'] = 10

    return firm_table


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


def extract_final_list_of_sector(firm_list: list):
    n = len(firm_list)
    present_sectors = list(set([firm.sector for firm in firm_list]))
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
    #household_table = gpd.read_file(filepath_region_table)
    #household_table = location_table.copy(deep=False)
    #TODO ajouter la demande finale par secteur country_sector_table
    #print(household_table)
    final_demand = sector_table["final_demand"]
    #household_table = household_table.stack() #.transpose()

    #TEST

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


def load_technical_coefficients(
        firm_list: list,
        filepath_tech_coef: str,
        io_cutoff: float = 0.1,
        import_sector_name: str | None = "IMP"
):
    """Load the input mix of the firms' Leontief function

    Parameters
    ----------
    firm_list : pandas.DataFrame
        the list of Firms generated from the createFirms function

    filepath_tech_coef : string
        Filepath to the matrix of technical coefficients

    io_cutoff : float
        Filters out technical coefficient below this cutoff value. Default to 0.1.

    import_sector_name : None or string
        Give the name of the import sector. If None, then the import technical coefficient is discarded. Default to None.

    Returns
    -------
    list of Firms
    """

    # Load technical coefficient matrix from data
    tech_coef_matrix = pd.read_csv(filepath_tech_coef, index_col=0)
    tech_coef_matrix = tech_coef_matrix.mask(tech_coef_matrix <= io_cutoff, 0)

    # We select only the technical coefficient between sectors that are actually represented in the economy
    # Note that, when filtering out small sector-district combination, some sector may not be present.
    sector_present = list(set([firm.sector for firm in firm_list]))
    if import_sector_name:
        tech_coef_matrix = tech_coef_matrix.loc[sector_present + [import_sector_name], sector_present]
    else:
        tech_coef_matrix = tech_coef_matrix.loc[sector_present, sector_present]

    # Check whether all sectors have input
    cond_sector_no_inputs = tech_coef_matrix.sum() == 0
    if cond_sector_no_inputs.any():
        warnings.warn(
            'Some sectors have no inputs: ' + str(cond_sector_no_inputs[cond_sector_no_inputs].index.to_list())
            + " Check this sector or reduce the io_coef cutoff")

    # Load input mix
    for firm in firm_list:
        firm.input_mix = tech_coef_matrix.loc[tech_coef_matrix.loc[:, firm.sector] != 0, firm.sector].to_dict()

    logging.info('Technical coefficient loaded. io_cutoff: ' + str(io_cutoff))

    return firm_list


def calibrate_input_mix(
        firm_list: str,
        firm_table: pd.DataFrame,
        sector_table: pd.DataFrame,
        filepath_transaction_table: str
):
    transaction_table = pd.read_csv(filepath_transaction_table)

    domestic_B2B_sales_per_firm = transaction_table.groupby('supplier_id')['transaction'].sum()
    firm_table['domestic_B2B_sales'] = firm_table['id'].map(domestic_B2B_sales_per_firm).fillna(0)
    firm_table['output'] = firm_table['domestic_B2B_sales'] + firm_table['final_demand'] + firm_table['exports']

    # Identify the sector of the products exchanged recorded in the transaction table and whether they are essential
    transaction_table['product_sector'] = transaction_table['supplier_id'].map(firm_table.set_index('id')['sector'])
    transaction_table['is_essential'] = transaction_table['product_sector'].map(
        sector_table.set_index('sector')['essential'])

    # Get input mix from this data
    def get_input_mix(transaction_from_unique_buyer, firm_tab):
        output = firm_tab.set_index('id').loc[transaction_from_unique_buyer.name, 'output']
        print(output)
        cond_essential = transaction_from_unique_buyer['is_essential']
        # for essential inputs, get total input per product type
        input_essential = transaction_from_unique_buyer[cond_essential].groupby('product_sector')[
                              'transaction'].sum() / output
        # for non essential inputs, get total input
        input_nonessential = transaction_from_unique_buyer.loc[~cond_essential, 'transaction'].sum() / output
        # get share how much is essential and evaluate how much can be produce with essential input only (beta)
        share_essential = input_essential.sum() / transaction_from_unique_buyer['transaction'].sum()
        max_output_with_essential_only = share_essential * output
        # shape results
        dic_res = input_essential.to_dict()
        dic_res['non_essential'] = input_nonessential
        dic_res['max_output_with_essential_only'] = max_output_with_essential_only
        return dic_res

    input_mix = transaction_table.groupby('buyer_id').apply(get_input_mix, firm_table)

    # Load input mix into Firms
    for firm in firm_list:
        firm.input_mix = input_mix[firm.pid]

    return firm_table, transaction_table


def load_inventories(firm_list: list, inventory_duration_target: int | str,
                     filepath_inventory_duration_targets: Path, extra_inventory_target: int | None = None,
                     inputs_with_extra_inventories: None | list = None,
                     buying_sectors_with_extra_inventories: None | list = None,
                     min_inventory: int = 1, random_mean_sd: float | None = None):
    """Load inventory duration target

    If inventory_duration_target is an integer, it is uniformly applied to all firms.
    If it its "inputed", then we use the targets defined in the file filepath_inventory_duration_targets. In that case,
    targets are sector specific, i.e., it varies according to the type of input and the sector of the buying firm.
    If both cases, we can add extra units of inventories:
    - uniformly, e.g., all firms have more inventories of all inputs,
    - to specific inputs, all firms have extra agricultural inputs,
    - to specific buying firms, e.g., all manufacturing firms have more of all inputs,
    - to a combination of both. e.g., all manufacturing firms have more of agricultural inputs.
    We can also add some noise on the distribution of inventories. Not yet imlemented.

    Parameters
    ----------
    firm_list : pandas.DataFrame
        the list of Firms generated from the createFirms function

    inventory_duration_target : "inputed" or integer
        Inventory duration target uniformly applied to all firms and all inputs.
        If 'inputed', uses the specific values from the file specified by
        filepath_inventory_duration_targets

    extra_inventory_target : None or integer
        If specified, extra inventory duration target.

    inputs_with_extra_inventories : None or list of sector
        For which inputs do we add inventories.

    buying_sectors_with_extra_inventories : None or list of sector
        For which sector we add inventories.

    min_inventory : int
        Set a minimum inventory level

    random_mean_sd: None
        Not yet implemented.

    Returns
    -------
        list of Firms
    """

    if isinstance(inventory_duration_target, int):
        for firm in firm_list:
            firm.inventory_duration_target = {input_sector: inventory_duration_target for input_sector in
                                              firm.input_mix.keys()}

    elif inventory_duration_target == 'inputed':
        dic_sector_inventory = \
            pd.read_csv(filepath_inventory_duration_targets).set_index(['buying_sector', 'input_sector'])[
                'inventory_duration_target'].to_dict()
        for firm in firm_list:
            firm.inventory_duration_target = {
                input_sector: dic_sector_inventory[(firm.sector, input_sector)]
                for input_sector in firm.input_mix.keys()
            }

    else:
        raise ValueError("Unknown value entered for 'inventory_duration_target'")

    # if random_mean_sd:
    #     if random_draw:
    #         for firm in firm_list:
    #             firm.inventory_duration_target = {}
    #             for input_sector in firm.input_mix.keys():
    #                 mean = dic_sector_inventory[(firm.sector, input_sector)]['mean']
    #                 sd = dic_sector_inventory[(firm.sector, input_sector)]['sd']
    #                 mu = math.log(mean/math.sqrt(1+sd**2/mean**2))
    #                 sigma = math.sqrt(math.log(1+sd**2/mean**2))
    #                 safety_day = np.random.log(mu, sigma)
    #                 firm.inventory_duration_target[input_sector] = safety_day

    # Add extra inventories if needed. Not the best programming maybe...
    if isinstance(extra_inventory_target, int):
        if isinstance(inputs_with_extra_inventories, list) and (buying_sectors_with_extra_inventories == 'all'):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector] + extra_inventory_target
                    if (input_sector in inputs_with_extra_inventories) else firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif (inputs_with_extra_inventories == 'all') and isinstance(buying_sectors_with_extra_inventories, list):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector] + extra_inventory_target
                    if (firm.sector in buying_sectors_with_extra_inventories) else firm.inventory_duration_target[
                        input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif isinstance(inputs_with_extra_inventories, list) and isinstance(buying_sectors_with_extra_inventories,
                                                                            list):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector] + extra_inventory_target
                    if ((input_sector in inputs_with_extra_inventories) and (
                            firm.sector in buying_sectors_with_extra_inventories)) else
                    firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif (inputs_with_extra_inventories == 'all') and (buying_sectors_with_extra_inventories == 'all'):
            for firm in firm_list:
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[input_sector] + extra_inventory_target
                    for input_sector in firm.input_mix.keys()
                }

        else:
            raise ValueError("Unknown value given for 'inputs_with_extra_inventories' or 'buying_sectors_with_extra_inventories'.\
                Should be a list of string or 'all'")

    if min_inventory > 0:
        for firm in firm_list:
            firm.inventory_duration_target = {
                input_sector: max(min_inventory, inventory)
                for input_sector, inventory in firm.inventory_duration_target.items()
            }
    # inventory_table = pd.DataFrame({
    #     'id': [firm.pid for firm in firm_list],
    #     'buying_sector': [firm.sector for firm in firm_list],
    #     'inventories': [firm.inventory_duration_target for firm in firm_list]
    # })
    # inventory_table.to_csv('inventory_check.csv')
    # logging.info("Inventories: "+str({firm.pid: firm.inventory_duration_target for firm in firm_list}))
    logging.info('Inventory duration targets loaded')
    if extra_inventory_target:
        logging.info("Extra inventory duration: " + str(extra_inventory_target) + \
                     " for inputs " + str(inputs_with_extra_inventories) + \
                     " for buying sectors " + str(buying_sectors_with_extra_inventories))
    return firm_list


def create_countries(filepath_imports: Path, filepath_exports: Path, filepath_transit: Path,
                     transport_nodes: geopandas.GeoDataFrame, present_sectors: list,
                     countries_to_include: list | str = 'all', time_resolution: str = "week",
                     target_units: str = "mUSD", input_units: str = "USD"):
    """Create the countries

    Parameters
    ----------
    filepath_imports : string
        path to import table csv

    filepath_exports : string
        path to export table csv

    filepath_transit : string
        path to transit matrix csv

    transport_nodes : pandas.DataFrame
        transport nodes

    present_sectors : list of string
        list which sectors are included. Output of the rescaleFirms functions.

    countries_to_include : list of string or "all"
        List of countries to include. Default to "all", which select all sectors.

    time_resolution : see rescaleMonetaryValues
    target_units : see rescaleMonetaryValues
    input_units : see rescaleMonetaryValues

    Returns
    -------
    list of Countries
    """
    logging.info('Creating country_list. Countries included: ' + str(countries_to_include))

    import_table = rescale_monetary_values(
        pd.read_csv(filepath_imports, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    export_table = rescale_monetary_values(
        pd.read_csv(filepath_exports, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    transit_matrix = rescale_monetary_values(
        pd.read_csv(filepath_transit, index_col=0),
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # entry_point_table = pd.read_csv(filepath_entry_points)

    # Keep only selected countries, if applicable
    if isinstance(countries_to_include, list):
        import_table = import_table.loc[countries_to_include, present_sectors]
        export_table = export_table.loc[countries_to_include, present_sectors]
        transit_matrix = transit_matrix.loc[countries_to_include, countries_to_include]
    elif countries_to_include == 'all':
        import_table = import_table.loc[:, present_sectors]
        export_table = export_table.loc[:, present_sectors]
    else:
        raise ValueError("'countries_to_include' should be a list of string or 'all'")

    logging.info("Total imports per " + time_resolution + " is " +
                 "{:.01f} ".format(import_table.sum().sum()) + target_units)
    logging.info("Total exports per " + time_resolution + " is " +
                 "{:.01f} ".format(export_table.sum().sum()) + target_units)
    logging.info("Total transit per " + time_resolution + " is " +
                 "{:.01f} ".format(transit_matrix.sum().sum()) + target_units)

    country_list = []
    total_imports = import_table.sum().sum()
    for country in import_table.index.tolist():
        # transit points
        # entry_points = entry_point_table.loc[
        #     entry_point_table['country']==country, 'entry_point'
        # ].astype(int).tolist()
        # odpoint
        cond_country = transport_nodes['special'] == country
        odpoint = transport_nodes.loc[cond_country, "id"]
        lon = transport_nodes.geometry.x
        lat = transport_nodes.geometry.y
        if len(odpoint) == 0:
            raise ValueError('No odpoint found for ' + country)
        elif len(odpoint) > 2:
            raise ValueError('More than 1 odpoint for ' + country)
        else:
            odpoint = odpoint.iloc[0]
            lon = lon.iloc[0]
            lat = lat.iloc[0]

        # imports, i.e., sales of countries
        qty_sold = import_table.loc[country, :]
        qty_sold = qty_sold[qty_sold > 0].to_dict()
        supply_importance = sum(qty_sold.values()) / total_imports

        # exports, i.e., purchases from countries
        qty_purchased = export_table.loc[country, :]
        qty_purchased = qty_purchased[qty_purchased > 0].to_dict()

        # transits
        # Note that transit are not given per sector, so, if we only consider a few sector, the full transit flows will still be used
        transit_from = transit_matrix.loc[:, country]
        transit_from = transit_from[transit_from > 0].to_dict()
        transit_to = transit_matrix.loc[country, :]
        transit_to = transit_to[transit_to > 0].to_dict()

        # create the list of Country object
        country_list += [Country(pid=country,
                                 qty_sold=qty_sold,
                                 qty_purchased=qty_purchased,
                                 odpoint=odpoint,
                                 long=lon,
                                 lat=lat,
                                 transit_from=transit_from,
                                 transit_to=transit_to,
                                 supply_importance=supply_importance
                                 )]
    country_list = CountryList(country_list)

    logging.info('Country_list created: ' + str([country.pid for country in country_list]))

    return country_list


def load_ton_usd_equivalence(sector_table: pd.DataFrame, firm_list: list, country_list: list):
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

    :return: (list(Firm objects), list(Country objects))
    """
    sector_to_usd_per_ton = sector_table.set_index('sector')['usd_per_ton']
    for firm in firm_list:
        firm.usd_per_ton = sector_to_usd_per_ton[firm.sector]
    for country in country_list:
        country.usd_per_ton = sector_to_usd_per_ton.mean()
    return firm_list, country_list, sector_to_usd_per_ton  # Should disappear, we want only sector_to_usdPerTon
