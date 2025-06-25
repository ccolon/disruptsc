import logging
from pathlib import Path
from typing import Dict, List, Union, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np

from disruptsc.agents.country import Country, Countries
from disruptsc.model.basic_functions import rescale_monetary_values, find_nearest_node_id
from disruptsc.network.mrio import Mrio


def _rescale_trade_matrix(matrix: pd.DataFrame, time_resolution: str, 
                         target_units: str, input_units: str) -> pd.DataFrame:
    """Rescale a trade matrix to target units and time resolution."""
    return rescale_monetary_values(
        matrix,
        input_time_resolution="year",
        target_time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )


def _validate_country_data(country_list: List[str], country_table: gpd.GeoDataFrame) -> None:
    """Validate that all countries have spatial data."""
    missing_countries = set(country_list) - set(country_table.index)
    if missing_countries:
        raise ValueError(f"Countries missing from spatial data: {missing_countries}")


def _extract_trade_matrices(mrio: Mrio, time_resolution: str, target_units: str, 
                           input_units: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract and rescale import, export, and transit matrices from MRIO."""
    # Get country lists
    buying_countries = mrio.external_buying_countries
    selling_countries = mrio.external_selling_countries
    
    # Extract and rescale import table
    import_table = _rescale_trade_matrix(
        mrio.loc[(selling_countries, mrio.import_label), mrio.region_sectors],
        time_resolution, target_units, input_units
    )
    import_table.index = [tup[0] for tup in import_table.index]
    import_table.columns = ['_'.join(tup) for tup in import_table.columns]
    
    # Extract and rescale export table
    export_table = _rescale_trade_matrix(
        mrio.loc[mrio.region_sectors, (buying_countries, mrio.export_label)],
        time_resolution, target_units, input_units
    )
    export_table.columns = [tup[0] for tup in export_table.columns]
    export_table.index = ['_'.join(tup) for tup in export_table.index]
    
    # Extract and rescale transit matrix
    transit_matrix = _rescale_trade_matrix(
        mrio.loc[(selling_countries, mrio.import_label), (buying_countries, mrio.export_label)],
        time_resolution, target_units, input_units
    )
    transit_matrix.columns = [tup[0] for tup in transit_matrix.columns]
    transit_matrix.index = [tup[0] for tup in transit_matrix.index]
    
    return import_table, export_table, transit_matrix


def _prepare_country_spatial_data(filepath_countries_spatial: Path, filepath_sectors: Path,
                                  country_list: List[str], transport_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load and prepare country spatial data."""
    # Load spatial data
    country_table = gpd.read_file(filepath_countries_spatial).set_index('region')
    
    # Validate data
    _validate_country_data(country_list, country_table)
    country_table = country_table.loc[country_list]
    
    # Find nearest transport nodes
    admissible_node_modes = ['roads', 'railways', 'maritime']
    potential_nodes = transport_nodes[transport_nodes['type'].isin(admissible_node_modes)]
    country_table['od_point'] = find_nearest_node_id(potential_nodes, country_table)
    
    # Add coordinate columns
    country_table['long'] = country_table["geometry"].x
    country_table['lat'] = country_table["geometry"].y
    
    # Add USD per ton data
    sector_data = pd.read_csv(filepath_sectors).set_index("sector")
    if 'IMP' not in sector_data.index:
        raise ValueError("Import sector 'IMP' not found in sector table")
    country_table['country_usd_per_ton'] = sector_data.loc['IMP', 'usd_per_ton']
    
    return country_table


def _create_country_trade_data(country: str, import_table: pd.DataFrame, export_table: pd.DataFrame,
                              transit_matrix: pd.DataFrame, buying_countries: List[str], 
                              selling_countries: List[str], total_imports: float) -> Dict:
    """Create trade data for a single country."""
    trade_data = {}
    
    # Process imports (country as seller)
    if country in selling_countries and total_imports > 0:
        qty_sold = import_table.loc[country, :]
        qty_sold = qty_sold[qty_sold > 0].to_dict()
        supply_importance = sum(qty_sold.values()) / total_imports
    else:
        qty_sold = {}
        supply_importance = 0
    
    # Process exports (country as buyer)
    if country in buying_countries:
        qty_purchased = export_table.loc[:, country]
        qty_purchased = qty_purchased[qty_purchased > 0].to_dict()
    else:
        qty_purchased = {}
    
    # Process transit flows
    transit_from = {}
    transit_to = {}
    
    if country in transit_matrix.columns:
        transit_from = transit_matrix.loc[:, country]
        transit_from = transit_from[transit_from > 0].to_dict()
    
    if country in transit_matrix.index:
        transit_to = transit_matrix.loc[country, :]
        transit_to = transit_to[transit_to > 0].to_dict()
    
    return {
        'qty_sold': qty_sold,
        'qty_purchased': qty_purchased,
        'transit_from': transit_from,
        'transit_to': transit_to,
        'supply_importance': supply_importance
    }


def create_countries_from_mrio(mrio: Mrio,
                               transport_nodes: gpd.GeoDataFrame,
                               filepath_countries_spatial: Path, filepath_sectors: Path,
                               time_resolution: str,
                               target_units: str, input_units: str) -> Tuple[Countries, gpd.GeoDataFrame]:
    """Create countries from MRIO data with trade flows and spatial information.
    
    Args:
        mrio: Multi-regional input-output data
        transport_nodes: Transport network nodes
        filepath_countries_spatial: Path to country spatial data
        filepath_sectors: Path to sector data
        time_resolution: Target time resolution
        target_units: Target monetary units
        input_units: Input monetary units
        
    Returns:
        Tuple of (Countries collection, country spatial table)
    """
    logging.info('Creating countries from MRIO data')
    
    # Extract countries from MRIO
    buying_countries = mrio.external_buying_countries
    selling_countries = mrio.external_selling_countries
    country_list = list(set(buying_countries) | set(selling_countries))
    logging.info(f'Found {len(country_list)} countries: {country_list}')
    
    # Extract and process trade matrices
    import_table, export_table, transit_matrix = _extract_trade_matrices(
        mrio, time_resolution, target_units, input_units
    )
    
    # Log trade totals
    total_imports = import_table.sum().sum()
    total_exports = export_table.sum().sum()
    total_transit = transit_matrix.sum().sum()
    
    logging.info(f"Total imports per {time_resolution}: {total_imports:.1f} {target_units}")
    logging.info(f"Total exports per {time_resolution}: {total_exports:.1f} {target_units}")
    logging.info(f"Total transit per {time_resolution}: {total_transit:.1f} {target_units}")
    
    # Prepare spatial data
    country_table = _prepare_country_spatial_data(
        filepath_countries_spatial, filepath_sectors, country_list, transport_nodes
    )
    
    # Create countries
    countries = Countries()
    for country in country_list:
        trade_data = _create_country_trade_data(
            country, import_table, export_table, transit_matrix,
            buying_countries, selling_countries, total_imports
        )
        
        # Create country object
        countries[country] = Country(
            pid=country,
            qty_sold=trade_data['qty_sold'],
            qty_purchased=trade_data['qty_purchased'],
            od_point=country_table.loc[country, "od_point"],
            long=country_table.loc[country, "long"],
            lat=country_table.loc[country, "lat"],
            transit_from=trade_data['transit_from'],
            transit_to=trade_data['transit_to'],
            usd_per_ton=country_table.loc[country, 'country_usd_per_ton'],
            supply_importance=trade_data['supply_importance'],
            import_label=mrio.import_label
        )
    
    logging.info(f'Created {len(countries)} countries: {countries.get_properties("pid")}')
    return countries, country_table
