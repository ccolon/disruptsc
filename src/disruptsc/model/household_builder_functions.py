import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd

from disruptsc.agents.household import Household, Households
from disruptsc.model.builder_functions import get_long_lat
from disruptsc.model.basic_functions import rescale_monetary_values, find_nearest_node_id
from disruptsc.network.mrio import Mrio


def create_households(
        household_table: pd.DataFrame,
        household_sector_consumption: dict[int, dict[str, float]]
) -> Households:
    """Create household agents from processed data.

    Takes household table and consumption data to create Household agent instances.
    Each household is assigned spatial coordinates and consumption patterns.

    Parameters
    ----------
    household_table : pd.DataFrame
        Table with household data including id, name, od_point, region, long, lat, population
    household_sector_consumption : dict[int, dict[str, float]]
        Household consumption by sector: {household_id: {sector: consumption_amount}}

    Returns
    -------
    Households
        Collection of Household agents ready for simulation
    """

    logging.debug('Creating households')
    household_table = household_table.set_index('id')
    households = Households([
        Household('hh_' + str(i),
                  name=household_table.loc[i, "name"],
                  od_point=household_table.loc[i, "od_point"],
                  region=household_table.loc[i, "region"],
                  long=float(household_table.loc[i, 'long']),
                  lat=float(household_table.loc[i, 'lat']),
                  population=household_table.loc[i, "population"],
                  sector_consumption=household_sector_consumption[i]
                  )
        for i in household_table.index.tolist()
    ])
    logging.info('Households generated')

    return households


def _load_and_assign_household_spatial_data(
    filepath_households_spatial: Path, 
    mrio: Mrio, 
    transport_nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Load household spatial data and assign to transport nodes.
    
    Parameters
    ----------
    filepath_households_spatial : Path
        Path to spatial household distribution file
    mrio : Mrio
        Multi-regional input-output table for region filtering
    transport_nodes : gpd.GeoDataFrame
        Transport network nodes
        
    Returns
    -------
    gpd.GeoDataFrame
        Household table with transport node assignments
    """
    # Load and filter spatial data
    household_table = gpd.read_file(filepath_households_spatial)
    household_table = household_table[household_table["region"].isin([tup[0] for tup in mrio.region_households])]
    
    # Assign to transport nodes
    admissible_node_mode = ['roads']
    potential_nodes = transport_nodes[transport_nodes['type'].isin(admissible_node_mode)]
    household_table['od_point'] = find_nearest_node_id(potential_nodes, household_table)
    
    logging.info(f"Select {household_table.shape[0]} households in {household_table['region'].nunique()} regions")
    return household_table


def _add_household_coordinates_and_identifiers(
    household_table: gpd.GeoDataFrame, 
    transport_nodes: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Add coordinates, IDs, and descriptive names to household table.
    
    Parameters
    ----------
    household_table : gpd.GeoDataFrame
        Household table with transport node assignments
    transport_nodes : gpd.GeoDataFrame
        Transport network nodes for coordinate lookup
        
    Returns
    -------
    pd.DataFrame
        Household table with coordinates and identifiers
    """
    # Add coordinates
    long_lat = get_long_lat(household_table['od_point'], transport_nodes)
    household_table['long'] = long_lat['long']
    household_table['lat'] = long_lat['lat']

    # Add IDs
    household_table['id'] = range(household_table.shape[0])
    household_table['household'] = "hh_" + household_table['id'].astype(str)

    # Add population default
    if "population" not in household_table.columns:
        household_table['population'] = 1

    # Add descriptive names (optimized)
    household_table['name'] = household_table.groupby('region').cumcount()
    household_table['name'] = (household_table['region'] + '_household' + 
                              household_table['name'].astype(str))
    
    return household_table


def _prepare_final_demand_data(
    mrio: Mrio,
    present_region_sectors: list[str],
    time_resolution: str,
    target_units: str,
    input_units: str
) -> pd.DataFrame:
    """Prepare and rescale final demand data from MRIO.
    
    Parameters
    ----------
    mrio : Mrio
        Multi-regional input-output table
    present_region_sectors : list[str]
        List of region_sector combinations to include
    time_resolution : str
        Target time resolution
    target_units : str
        Target monetary units
    input_units : str
        Input monetary units in MRIO data
        
    Returns
    -------
    pd.DataFrame
        Rescaled final demand data
    """
    present_import_countries = [(country + '_' + mrio.import_label) 
                               for country in mrio.external_selling_countries]
    final_demand = mrio.get_final_demand(present_region_sectors + present_import_countries)
    
    final_demand = rescale_monetary_values(
        final_demand,
        input_time_resolution="year",
        target_time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    
    return final_demand


def _calculate_household_consumption_patterns(
    household_table: pd.DataFrame,
    final_demand: pd.DataFrame,
    final_demand_cutoff: dict,
    time_resolution: str,
    target_units: str
) -> dict[int, dict[str, float]]:
    """Calculate consumption patterns for each household.
    
    Distributes final demand across households based on population proportions
    within each region.
    
    Parameters
    ----------
    household_table : pd.DataFrame
        Household table with population data
    final_demand : pd.DataFrame
        Final demand data by region and sector
    final_demand_cutoff : dict
        Cutoff configuration for minimum consumption
    time_resolution : str
        Target time resolution
    target_units : str
        Target monetary units
        
    Returns
    -------
    dict[int, dict[str, float]]
        Household consumption by sector: {household_id: {sector: consumption}}
    """
    # Calculate population proportions per region
    total_population_per_region = household_table.groupby('region')['population'].sum()
    
    # Prepare cutoff threshold
    cutoff = rescale_monetary_values(
        final_demand_cutoff['value'],
        input_time_resolution="year",
        target_time_resolution=time_resolution,
        target_units=target_units,
        input_units=final_demand_cutoff['unit']
    )
    
    # Calculate consumption for each household
    household_sector_consumption = {}
    
    for _, household in household_table.iterrows():
        # Population proportion within region
        pop_proportion = household['population'] / total_population_per_region[household['region']]
        
        # Get demand for this region
        if household['region'] in final_demand.columns.get_level_values(0):
            region_demand = final_demand.xs(household['region'], axis=1, level=0)
            
            # Scale demand for this household
            household_demand = (region_demand * pop_proportion).stack().to_dict()
            
            # Apply cutoff and store
            household_sector_consumption[household['id']] = {
                tup[0] + "_" + tup[1]: demand
                for tup, demand in household_demand.items()
                if demand > cutoff
            }
    
    return household_sector_consumption


def define_households_from_mrio(
        mrio: Mrio,
        filepath_households_spatial: Path,
        transport_nodes: gpd.GeoDataFrame,
        time_resolution: str,
        target_units: str,
        input_units: str,
        final_demand_cutoff: dict,
        present_region_sectors: list[str]
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """Define households from MRIO data and spatial information.
    
    Main orchestrator function that coordinates household creation by delegating
    to specialized helper functions for each phase.
    
    Parameters
    ----------
    mrio : Mrio
        Multi-regional input-output table containing final demand data
    filepath_households_spatial : Path
        Path to spatial household distribution file (GeoJSON/Shapefile)
    transport_nodes : gpd.GeoDataFrame
        Transport network nodes for household assignment
    time_resolution : str
        Target time resolution for consumption data (e.g., 'day', 'week', 'month')
    target_units : str
        Target monetary units for consumption (e.g., 'USD', 'kUSD')
    input_units : str
        Input monetary units in MRIO data
    final_demand_cutoff : dict
        Cutoff configuration for minimum consumption thresholds
    present_region_sectors : list[str]
        List of region_sector combinations to include
        
    Returns
    -------
    tuple[pd.DataFrame, dict[int, dict[str, float]]]
        Tuple of (household_table, household_sector_consumption)
        - household_table: DataFrame with household spatial and demographic data
        - household_sector_consumption: Dict mapping household_id to sector consumption
    """
    # 1. Load spatial data and assign to transport nodes
    household_table = _load_and_assign_household_spatial_data(
        filepath_households_spatial, mrio, transport_nodes
    )
    
    # 2. Add coordinates, IDs, and names
    household_table = _add_household_coordinates_and_identifiers(household_table, transport_nodes)
    
    # 3. Prepare final demand data
    final_demand = _prepare_final_demand_data(
        mrio, present_region_sectors, time_resolution, target_units, input_units
    )
    
    # 4. Calculate household consumption patterns
    household_sector_consumption = _calculate_household_consumption_patterns(
        household_table, final_demand, final_demand_cutoff, time_resolution, target_units
    )
    
    logging.info(f"Created {household_table.shape[0]} households in {household_table['od_point'].nunique()} od points")
    
    return household_table, household_sector_consumption


