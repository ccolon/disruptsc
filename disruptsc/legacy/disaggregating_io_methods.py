"""
Disaggregating IO Methods Archive
=================================

This file contains all the methods and functions that were specific to the 
"disaggregating IO" firm_data_type mode, which was removed from DisruptSC.

These methods are preserved here for reference and potential future use.

Original files:
- disruptsc/model/firm_builder_functions.py (define_firms_from_local_economic_data)
- disruptsc/model/input_validation.py (_validate_disaggregating_io_inputs, _validate_region_data)
- disruptsc/model/household_builder_functions.py (regional data integration)

Archived on: December 6, 2024
Version: 1.0.8
"""

import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas
import pandas
import pandas as pd
import geopandas as gpd
import numpy as np

# These imports would need to be updated if restoring this code
# from disruptsc.model.basic_functions import find_nearest_node_id
# from disruptsc.model.builder_functions import get_index_closest_point, get_long_lat, get_absolute_cutoff_value


def define_firms_from_local_economic_data(filepath_region_economic_data: Path,
                                          sectors_to_include: list, transport_nodes: geopandas.GeoDataFrame,
                                          filepath_sector_table: Path, min_nb_firms_per_sector: int):
    """Define firms based on the region_economic_data.
    The output is a dataframe, 1 row = 1 firm.
    The instances Firms are created in the createFirm function.

    Steps:
    1. Load the region_economic_data
    2. It adds a row only when the sector in one region is higher than the sector_cutoffs
    3. It identifies the node of the road network that is the closest to the region point
    4. It combines the firms of the same sector that are in the same road node (case of 2 regions close
    to the same road node)
    5. It calculates the "importance" of each firm = their size relative to the sector size

    Parameters
    ----------
    min_nb_firms_per_sector
    filepath_region_economic_data: string
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
    region_eco_data = gpd.read_file(filepath_region_economic_data)
    sector_table = pd.read_csv(filepath_sector_table)

    # A.2. for each sector, select region where supply_data is over threshold
    # and populate firm table
    firm_table_per_region = pd.DataFrame()
    for sector, row in sector_table.set_index("sector").iterrows():
        if (sectors_to_include == "all") or (sector in sectors_to_include):
            # check that the supply metric is in the data
            if row["supply_data"] not in region_eco_data.columns:
                logging.warning(f"{row['supply_data']} for sector {sector} is missing from the economic data. "
                                f"We will create by default firms in the {min_nb_firms_per_sector} "
                                f"most populated regions")
                where_create_firm = region_eco_data["population"].nlargest(min_nb_firms_per_sector).index
                # populate firm table
                new_firm_table = pd.DataFrame({
                    "sector": sector,
                    "region": region_eco_data.loc[where_create_firm, "region"].tolist(),
                    "population": region_eco_data.loc[where_create_firm, "population"].tolist(),
                    "absolute_size": region_eco_data.loc[where_create_firm, "population"].tolist()
                })
            else:
                # create one firm where economic metric is over threshold
                where_create_firm = region_eco_data[row["supply_data"]] > row["cutoff"]
                # if it results in less than 5 firms, we go below the cutoff to get at least 5 firms,
                # only if there are enough regions with positive supply_data
                if where_create_firm.sum() < min_nb_firms_per_sector:
                    cond_positive_supply_data = region_eco_data[row["supply_data"]] > 0
                    where_create_firm = region_eco_data.loc[cond_positive_supply_data, row["supply_data"]].nlargest(
                        min_nb_firms_per_sector).index
                # populate firm table
                new_firm_table = pd.DataFrame({
                    "sector": sector,
                    "region": region_eco_data.loc[where_create_firm, "region"].tolist(),
                    "population": region_eco_data.loc[where_create_firm, "population"].tolist(),
                    "absolute_size": region_eco_data.loc[where_create_firm, row["supply_data"]]
                })

            new_firm_table['relative_size'] = new_firm_table['absolute_size'] / new_firm_table['absolute_size'].sum()
            firm_table_per_region = pd.concat([firm_table_per_region, new_firm_table], axis=0)

    # B. Assign firms to the closest road nodes
    # B.1. Create a dictionary that link a region to id of the closest road node
    # Create dic that links regions to points
    selected_regions = list(firm_table_per_region['region'].unique())
    logging.info('Select ' + str(firm_table_per_region.shape[0]) +
                 " in " + str(len(selected_regions)) + ' regions')
    cond = region_eco_data['region'].isin(selected_regions)
    logging.info('Assigning firms to od-points')
    dic_selected_region_to_points = region_eco_data[cond].set_index('region')['geometry'].to_dict()
    # Select road node points
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    # Create dic - NOTE: get_index_closest_point would need to be imported
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_selected_region_to_points.items()
    }

    # B.2. Map firm to the closest road node
    firm_table_per_region['od_point'] = firm_table_per_region['region'].map(dic_region_to_road_node_id)

    # C. Combine firms that are in the same od-point and in the same sector
    # group by od-point and sector
    firm_table_per_od_point = firm_table_per_region \
        .groupby(['region', 'od_point', 'sector'], as_index=False) \
        .sum()

    # D. Add information required by the createFirms function
    # add sector type
    sector_to_sector_type = sector_table.set_index('sector')['type']
    firm_table_per_od_point['sector_type'] = firm_table_per_od_point['sector'].map(sector_to_sector_type)
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table_per_od_point['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table_per_od_point['long'] = firm_table_per_od_point['od_point'].map(road_node_id_to_long_lat['long'])
    firm_table_per_od_point['lat'] = firm_table_per_od_point['od_point'].map(road_node_id_to_long_lat['lat'])
    # add id
    firm_table_per_od_point['id'] = list(range(firm_table_per_od_point.shape[0]))
    # add name, not really useful
    firm_table_per_od_point['name'] = firm_table_per_od_point['od_point'].astype(str) + '-' + firm_table_per_od_point[
        'sector']
    # add importance
    firm_table_per_od_point['importance'] = firm_table_per_od_point['relative_size']

    # E. Print summary information
    cond = region_eco_data['region'].isin(selected_regions)
    represented_pop = region_eco_data.loc[cond, 'population'].sum()
    total_population = region_eco_data['population'].sum()

    logging.info("Summary of firm creation from regional data:")
    for sector, row in sector_table.set_index("sector").iterrows():
        if (sectors_to_include == "all") or (sector in sectors_to_include):
            if row["supply_data"] in region_eco_data.columns:
                logging.info("Sector " + sector + " represented " +
                             "{:.0f}%".format(firm_table_per_od_point[firm_table_per_od_point['sector'] == sector][
                                                  'absolute_size'].sum()
                                              / region_eco_data[row['supply_data']].sum() * 100) +
                             " of " + row['supply_data'] + " and " +
                             "{:.0f}%".format(firm_table_per_od_point[firm_table_per_od_point['sector'] == sector][
                                                  'population'].sum()
                                              / region_eco_data["population"].sum() * 100) +
                             " of population")

    return firm_table_per_od_point, firm_table_per_region


# Input Validation Methods for Disaggregating IO Mode
# ===================================================

def _validate_disaggregating_io_inputs(parameters):
    """Validate inputs specific to disaggregating IO mode.
    
    This was called from InputValidator._validate_all_inputs() when
    parameters.firm_data_type == "disaggregating IO"
    """
    errors = []
    warnings = []
    
    # Technical coefficients file is mandatory
    tech_coef_file = parameters.filepaths.get('tech_coef')
    if not tech_coef_file:
        errors.append("Disaggregating IO mode requires 'tech_coef' filepath to be specified in parameters")
    elif not Path(tech_coef_file).exists():
        errors.append(f"Required technical coefficients file not found: {tech_coef_file}")
    else:
        errors_tc, warnings_tc = _validate_tech_coef_file(tech_coef_file)
        errors.extend(errors_tc)
        warnings.extend(warnings_tc)
    
    # Regional economic data is mandatory
    region_data_file = parameters.filepaths.get('region_data')
    if not region_data_file:
        errors.append("Disaggregating IO mode requires 'region_data' filepath to be specified in parameters")
    elif not Path(region_data_file).exists():
        errors.append(f"Required regional data file not found: {region_data_file}")
    else:
        errors_rd, warnings_rd = _validate_region_data(region_data_file, parameters)
        errors.extend(errors_rd)
        warnings.extend(warnings_rd)
    
    return errors, warnings


def _validate_tech_coef_file(filepath):
    """Validate technical coefficients file structure."""
    errors = []
    warnings = []
    
    try:
        df = pd.read_csv(filepath, index_col=0)
    except Exception as e:
        errors.append(f"Cannot read technical coefficients file: {e}")
        return errors, warnings
    
    # Check if it's square
    if df.shape[0] != df.shape[1]:
        errors.append(f"Technical coefficients matrix must be square: {df.shape[0]} rows vs {df.shape[1]} columns")
    
    # Check for negative values (warnings, not errors)
    if (df < 0).any().any():
        warnings.append("Technical coefficients matrix contains negative values")
    
    # Check if coefficients are reasonable (< 1.0)
    if (df > 1.0).any().any():
        warnings.append("Technical coefficients matrix contains values > 1.0 - check units")
    
    return errors, warnings


def _validate_region_data(filepath, parameters):
    """Validate regional economic data file.
    
    This was called for disaggregating IO mode to ensure regional
    data had the required structure for spatial disaggregation.
    """
    errors = []
    warnings = []
    
    try:
        gdf = gpd.read_file(filepath)
    except Exception as e:
        errors.append(f"Cannot read region_data.geojson: {e}")
        return errors, warnings
    
    # Check required columns
    required_cols = ['admin_code', 'region', 'population']
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    if missing_cols:
        errors.append(f"region_data.geojson missing required columns: {missing_cols}")
    
    # Check geometry types
    if not all(gdf.geometry.geom_type == 'Point'):
        errors.append("region_data.geojson: All geometries must be Points")
    
    # Check for supply_data columns referenced in sector_table
    sector_table_file = parameters.filepaths.get('sector_table')
    if sector_table_file and Path(sector_table_file).exists():
        try:
            sector_df = pd.read_csv(sector_table_file)
            if 'supply_data' in sector_df.columns:
                supply_data_cols = sector_df['supply_data'].dropna().unique()
                missing_supply_cols = [col for col in supply_data_cols if col not in gdf.columns]
                if missing_supply_cols:
                    errors.append(f"region_data.geojson missing supply_data columns: {missing_supply_cols}")
        except Exception:
            warnings.append("Could not cross-validate supply_data columns with sector_table")
    
    # Check for negative population
    if 'population' in gdf.columns:
        if (gdf['population'] < 0).any():
            warnings.append("region_data.geojson contains negative population values")
        
        if gdf['population'].isna().any():
            warnings.append("region_data.geojson contains missing population values")
    
    return errors, warnings


# Household Builder Methods for Regional Data Integration
# ======================================================

def create_households_from_regional_data(
        mrio, 
        filepath_region_data: str,
        cutoff_household_demand: dict,
        combine_sector_cutoff: str,
        pop_density_cutoff: float,
        pop_cutoff: float,
        local_demand_cutoff: float,
        sectors_to_include: str,
        monetary_units_in_data: str,
        admin: list | None = None
):
    """Create households based on regional population and demand data.
    
    This method was used in disaggregating IO mode to place households
    based on regional economic data rather than MRIO final demand.
    
    Originally from household_builder_functions.py
    """
    
    # B1. Load region data to get tot population
    region_data = gpd.read_file(filepath_region_data)
    tot_pop = region_data['population'].sum()

    # Apply population and density filters
    cond = region_data['population'] > 0
    if pop_density_cutoff > 0:
        cond = cond & (region_data['pop_density'] >= pop_density_cutoff)
    if pop_cutoff > 0:
        cond = cond & (region_data['population'] >= pop_cutoff)
    
    household_table = region_data.loc[cond, ['population', 'geometry', 'region']].copy()
    
    logging.info(f"{cond.sum()} regions selected over {region_data.shape[0]} representing "
                 f"{(household_table['population'].sum() / region_data['population'].sum() * 100):.0f}% of population")

    # Calculate relative population for demand allocation
    rel_pop = household_table['population'] / region_data['population'].sum()
    
    # Add demand columns for each sector based on MRIO final demand
    final_demand_total = mrio.get_final_demand().sum(axis=1)
    
    for region_sector in mrio.region_sectors:
        if sectors_to_include == "all" or region_sector[1] in sectors_to_include:
            sector_final_demand = final_demand_total.get(region_sector, 0)
            household_table[f"demand_{region_sector[1]}"] = rel_pop * sector_final_demand
    
    # Apply local demand cutoff
    if local_demand_cutoff > 0:
        demand_cols = [col for col in household_table.columns if col.startswith('demand_')]
        for col in demand_cols:
            household_table.loc[household_table[col] < local_demand_cutoff, col] = 0
    
    # Add administrative level data if specified
    if admin:
        for admin_level in admin:
            if admin_level in region_data.columns:
                region_to_admin = region_data.set_index('region')[admin_level].to_dict()
                household_table[admin_level] = household_table['region'].map(region_to_admin)
    
    return household_table


# Summary of Archived Components
# =============================

ARCHIVED_FUNCTIONS = [
    "define_firms_from_local_economic_data",
    "_validate_disaggregating_io_inputs", 
    "_validate_tech_coef_file",
    "_validate_region_data",
    "create_households_from_regional_data"
]

ORIGINAL_FILES = {
    "define_firms_from_local_economic_data": "disruptsc/model/firm_builder_functions.py",
    "_validate_disaggregating_io_inputs": "disruptsc/model/input_validation.py",
    "_validate_region_data": "disruptsc/model/input_validation.py", 
    "create_households_from_regional_data": "disruptsc/model/household_builder_functions.py"
}

REMOVAL_DATE = "2024-12-06"
ARCHIVE_REASON = "Simplifying codebase to focus on MRIO and supplier-buyer network modes"

if __name__ == "__main__":
    print("Disaggregating IO Methods Archive")
    print("=================================")
    print(f"Contains {len(ARCHIVED_FUNCTIONS)} archived functions")
    print(f"Removed on: {REMOVAL_DATE}")
    print(f"Reason: {ARCHIVE_REASON}")
    print("\nArchived functions:")
    for func in ARCHIVED_FUNCTIONS:
        print(f"  - {func} (from {ORIGINAL_FILES.get(func, 'unknown')})")