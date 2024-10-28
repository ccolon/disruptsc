import logging
from pathlib import Path

import pandas
import pandas as pd
import geopandas as gpd
import numpy as np

from src.agents.household import Household, Households
from src.model.builder_functions import get_index_closest_point, get_long_lat, \
    get_closest_road_nodes
from src.model.basic_functions import rescale_monetary_values
from src.network.mrio import Mrio


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

    logging.debug('Creating households')
    household_table = household_table.set_index('id')
    households = Households([
        Household('hh_' + str(i),
                  name=household_table.loc[i, "name"],
                  od_point=household_table.loc[i, "od_point"],
                  long=float(household_table.loc[i, 'long']),
                  lat=float(household_table.loc[i, 'lat']),
                  population=household_table.loc[i, "population"],
                  sector_consumption=household_sector_consumption[i]
                  )
        for i in household_table.index.tolist()
    ])
    logging.info('Households generated')

    return households


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
    # Adrien's function for Global model
    # household_table = gpd.read_file(filepath_region_table)
    # household_table = location_table.copy(deep=False)
    # TODO ajouter la demande finale par secteur country_sector_table
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
    household_table.columns = ['region', 'column', 'value']

    ## Fin TEST

    # C. Create one household per OD point
    logging.info('Assigning households to od-points')
    dic_select_region_to_points = household_table.set_index('region')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_select_region_to_points.items()
    }
    # Map household to closest road nodes
    household_table['od_point'] = household_table['region'].map(dic_region_to_road_node_id)

    # Combine households that are in the same od-point
    household_table = household_table \
        .drop(columns=['geometry']) \
        .groupby(['region', 'od_point'], as_index=False) \
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
    mrio = Mrio.load_mrio_from_filepath(filepath_mrio)
    # mrio = pd.read_csv(filepath_mrio, header=[0, 1], index_col=[0, 1])
    # # Extract region_households
    # region_households = mrio.columns.to_list()
    # region_households = [tup for tup in region_households if tup[1] == 'final_demand']

    # Create household table
    household_table = pd.DataFrame({
        'tuple': mrio.region_households,
        'region': [tup[0] for tup in mrio.region_households],
        'name': [tup[0] + '_households' for tup in mrio.region_households]
    })
    # household_table = pd.DataFrame({"name": region_households})
    # household_table['region'] = household_table['name'].str.extract('([0-9]*)-H')
    # household_table['region'] = household_table['name'].str.extract('([A-Z]{3})_H')
    logging.info(f"Select {household_table.shape[0]} households in {household_table['region'].nunique()} regions")

    # Identify OD point
    household_table['od_point'] = get_closest_road_nodes(household_table['region'], transport_nodes,
                                                         filepath_region_table)

    # Add long lat
    long_lat = get_long_lat(household_table['od_point'], transport_nodes)
    household_table['long'] = long_lat['long']
    household_table['lat'] = long_lat['lat']

    # Add id
    household_table['id'] = range(household_table.shape[0])

    # Add population
    region_table = gpd.read_file(filepath_region_table)
    household_table['population'] = household_table['region'].map(region_table.set_index('region')['population'])

    # Identify final demand per region_sector
    consumption = mrio[mrio.region_households].copy(deep=True)   # TODO to put in the class Mrio
    consumption.index = ['_'.join(tup) for tup in consumption.index]
    consumption.columns = [tup[0] + '_households' for tup in mrio.region_households]
    consumption.columns = consumption.columns.map(household_table.set_index('name')['id'])
    consumption = rescale_monetary_values(
        consumption,
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # col_final_demand = [col for col in mrio.columns if col[-2:] == "-H"]  #TODO too specific
    # col_final_demand = [col for col in mrio.columns if col[-2:] == "_H"]  #TODO too specific
    # household_sector_consumption = mrio[col_final_demand].stack().reset_index() \
    #     .groupby('level_1') \
    #     .apply(lambda df: df.set_index('level_0')[0].to_dict()) \
    #     .to_dict()
    # Replace name by id :TODO use name as id to makes this unecessary
    # dic_name_to_id = household_table.set_index('name')['id']
    household_sector_consumption = consumption.to_dict()
    household_sector_consumption = {
        household: {
            key: value
            for key, value in sublist.items()
            if value > 0
        }
        for household, sublist in household_sector_consumption.items()
    }

    # Info
    logging.info(f"Create {household_table.shape[0]} households in {household_table['od_point'].nunique()} od points")

    return household_table, household_sector_consumption


def define_households(
        sector_table: pd.DataFrame,
        filepath_region_data: Path,
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
    filepath_region_data
    sector_table

    Returns
    -------
    household_table
    household_purchase_plan

    '''
    # A. Filter regions based on density
    # load file
    region_data = gpd.read_file(filepath_region_data)
    # filter & keep household were firms are
    cond = region_data['population'] > 0
    if pop_density_cutoff > 0:
        cond = cond & (region_data['pop_density'] >= pop_density_cutoff)
    if pop_cutoff > 0:
        cond = cond & (region_data['population'] >= pop_cutoff)
    # create household_table
    household_table = region_data.loc[cond, ['population', 'geometry', 'region']].copy()
    logging.info(f"{cond.sum()} regions selected over {region_data.shape[0]} representing "
                 f"{(household_table['population'].sum() / region_data['population'].sum() * 100):.0f}% of population")

    # B. Add final demand
    # Add imports... weird filtering: add if larger than the lowest sectoral final_demand filtered
    min_final_demand_filtered = sector_table.set_index('sector').loc[filtered_sectors, 'final_demand'].min()
    if sector_table.set_index('sector').loc['IMP', 'final_demand'] > min_final_demand_filtered:
        sectors_to_buy_from = filtered_sectors + ['IMP']
    else:
        sectors_to_buy_from = filtered_sectors
    # get final demand for the selected sector
    final_demand = sector_table.loc[sector_table['sector'].isin(sectors_to_buy_from), ['sector', 'final_demand']]

    # put as single row
    final_demand_as_row = final_demand.set_index('sector').transpose()
    # duplicates rows
    final_demand_each_household = pd.concat([final_demand_as_row for i in range(household_table.shape[0])])
    # align index and concat
    final_demand_each_household.index = household_table.index
    # compute final demand per region
    rel_pop = household_table['population'] / region_data['population'].sum()
    final_demand_each_household = final_demand_each_household.multiply(rel_pop, axis='index')
    # add to household table
    household_table = pd.concat([household_table, final_demand_each_household], axis=1)

    # C. Create one household per OD point
    logging.info('Assigning households to od-points')
    dic_select_region_to_points = household_table.set_index('region')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_select_region_to_points.items()
    }
    # Map household to the closest road nodes
    household_table['od_point'] = household_table['region'].map(dic_region_to_road_node_id)
    # Combine households that are in the same od-point
    household_table = household_table \
        .drop(columns=['geometry']) \
        .groupby(['od_point', 'region'], as_index=False) \
        .sum()
    logging.info(str(household_table.shape[0]) + ' od-point selected for demand')

    # D. Filter out small demand
    if local_demand_cutoff > 0:
        household_table[sectors_to_buy_from] = household_table[sectors_to_buy_from].mask(
            household_table[sectors_to_buy_from] < local_demand_cutoff
        )
    # info
    logging.info(f"Create {household_table.shape[0]} households in {household_table['od_point'].nunique()} od points")
    for sector in sectors_to_buy_from:
        logging.info(f"Sector {sector}: create {(~household_table[sector].isnull()).sum()} households that covers " +
                     "{:.0f}%".format(
                         household_table[sector].sum() \
                         / sector_table.set_index('sector').loc[sector, 'final_demand'] * 100
                     ) + " of total final demand"
                     )
    if (household_table[sectors_to_buy_from].sum(axis=1) == 0).any():
        logging.warning('Some households have no purchase plan because of all their sectoral demand below cutoff!')

    # E. Add information required by the createHouseholds function
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(household_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    household_table['long'] = household_table['od_point'].map(road_node_id_to_long_lat['long'])
    household_table['lat'] = household_table['od_point'].map(road_node_id_to_long_lat['lat'])
    # add id
    household_table['id'] = list(range(household_table.shape[0]))
    # # add name, not really useful
    household_table['name'] = "hh_" + household_table['od_point'].astype(str)

    # F. Create purchase plan per household
    # rescale according to time resolution
    household_table[sectors_to_buy_from] = rescale_monetary_values(
        household_table[sectors_to_buy_from],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    # to dict
    household_sector_consumption = household_table.set_index('id')[sectors_to_buy_from].to_dict(orient='index')
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


def add_households_for_firms(
        firm_table: pd.DataFrame,
        household_table: pd.DataFrame,
        filepath_region_data: str,
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
    added_household_table = added_household_table.merge(od_point_long_lat, how='left', on='od_point')

    # B1. Load region data to get tot population
    region_data = gpd.read_file(filepath_region_data)
    tot_pop = region_data['population'].sum()

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
                 str(household_table['od_point'].nunique()) + ' od points')
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
