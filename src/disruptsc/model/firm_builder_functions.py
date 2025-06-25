import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from disruptsc.agents.firm import Firm, Firms
from disruptsc.model.basic_functions import find_nearest_node_id
from disruptsc.model.builder_functions import get_index_closest_point, get_long_lat, get_absolute_cutoff_value
from disruptsc.network.mrio import Mrio


def create_firms(
        firm_table: pd.DataFrame,
        keep_top_n_firms: Optional[int] = None,
        inventory_restoration_time: float = 4,
        utilization_rate: float = 0.8,
        capital_to_value_added_ratio: float = 4,
        admin: Optional[list[str]] = None
) -> Firms:
    """Create the firms

    It uses firm_table from rescaleNbFirms

    Parameters
    ----------
    admin
    capital_to_value_added_ratio
    firm_table: pandas.DataFrame
        firm_table from rescaleNbFirms
    keep_top_n_firms: None (default) or integer
        (optional) can be specified if we want to keep only the first n firms, for testing purposes
    inventory_restoration_time: float
        Determines the speed at which firms try to reach their inventory duration target
    utilization_rate: float
        Set the utilization rate, which determines the production capacity at the input-output equilibrium.

    Returns
    -------
    list of Firms
    """

    if isinstance(keep_top_n_firms, int):
        firm_table = firm_table.iloc[:keep_top_n_firms, :]

    logging.debug('Creating firms')
    ids = firm_table['id'].tolist()
    firm_table = firm_table.set_index('id')

    firms = Firms([
        Firm(pid=i,
             region_sector=firm_table.loc[i, "region_sector"],
             region=firm_table.loc[i, "region"],
             sector_type=firm_table.loc[i, "sector_type"],
             sector=firm_table.loc[i, "sector"],
             od_point=firm_table.loc[i, "od_point"],
             usd_per_ton=firm_table.loc[i, "usd_per_ton"],
             importance=firm_table.loc[i, 'importance'],
             name=firm_table.loc[i, 'name'],
             long=float(firm_table.loc[i, 'long']),
             lat=float(firm_table.loc[i, 'lat']),
             target_margin=float(firm_table.loc[i, 'margin']),
             transport_share=float(firm_table.loc[i, 'transport_share']),
             utilization_rate=utilization_rate,
             inventory_restoration_time=inventory_restoration_time,
             capital_to_value_added_ratio=capital_to_value_added_ratio
             )
        for i in ids
    ])
    # We add a bit of noise to the long and lat coordinates
    # It allows to visually disentangle firms located at the same od-point when plotting the map.
    for firm in firms.values():
        firm.add_noise_to_geometry()

    if isinstance(admin, list):
        for pid, firm in firms.items():
            for admin_level in admin:
                value = firm_table.loc[pid, admin_level]
                setattr(firm, admin_level, value)

    return firms


def check_successful_extraction(firm_table: pd.DataFrame, attribute: str) -> None:
    if firm_table[attribute].isnull().sum() > 0:
        logging.warning(f"Unsuccessful extraction of {attribute} for "
                        f"{firm_table[firm_table[attribute].isnull()].index}")


def load_firms_spatial_data(filepath_firms_spatial: Path, accepted_sectors: list[str],
                            accepted_regions: list[str]) -> gpd.GeoDataFrame:
    """
    Load firm spatial data from single firms.geojson file.
    Replaces the old load_disag_data() that read from Disag/ folder.

    Parameters
    ----------
    accepted_regions : list
        List of regions to include
    accepted_sectors : list  
        List of sectors to include
    filepath_firms_spatial : Path
        Path to the firms.geojson file

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing firm spatial data
    """
    logging.info(f'Loading firm spatial data from {filepath_firms_spatial}')

    # Load the firms geojson file
    gdf = gpd.read_file(filepath_firms_spatial)

    # Validate structure
    if 'region' not in gdf.columns:
        raise ValueError("firms.geojson must have 'region' column")
    if not gdf.geometry.geom_type.eq('Point').all():
        raise ValueError("firms.geojson: All geometries must be Points")

    # Filter regions
    useless_regions = list(set(gdf['region'].unique()) - set(accepted_regions))
    if useless_regions:
        logging.info(f"The following regions will not be used: {useless_regions} because "
                     f"they are not part of the MRIO table: {accepted_regions}")
    gdf = gdf[~gdf['region'].isin(useless_regions)]

    # Filter columns to keep only relevant sectors
    useless_cols = [col for col in gdf.columns if col not in ['region', "geometry"] + accepted_sectors]
    if useless_cols:
        logging.info(f"The following columns will not be used: {useless_cols} because "
                     f"they are not part of the list of sectors defined in the MRIO table: {accepted_sectors}")
    disag_sectors = list(set(gdf.columns) & set(accepted_sectors))
    gdf = gdf[['region', 'geometry'] + disag_sectors]

    return gdf


def create_disag_firm_table(disag_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    disag_firm_table = disag_data.reset_index().drop("geometry", axis=1) \
        .set_index(['index', 'region']).stack().reset_index()
    disag_firm_table['geometry'] = disag_firm_table['index'].map(disag_data['geometry'])
    disag_firm_table = disag_firm_table.drop('index', axis=1)
    disag_firm_table.columns = ['region', 'sector', 'importance', 'geometry']
    disag_firm_table = disag_firm_table[disag_firm_table['importance'].notnull()]
    disag_firm_table = disag_firm_table[disag_firm_table['importance'] > 0]
    disag_firm_table['tuple'] = list(zip(disag_firm_table['region'], disag_firm_table['sector']))
    disag_firm_table['region_sector'] = disag_firm_table['region'] + '_' + disag_firm_table['sector']
    return gpd.GeoDataFrame(disag_firm_table, crs=disag_data.crs)


def _create_base_firm_table_from_mrio(mrio: Mrio, households_spatial: Path) -> gpd.GeoDataFrame:
    """Create base firm table from MRIO data with geometry from households.
    
    Parameters
    ----------
    mrio : Mrio
        Multi-regional input-output table
    households_spatial : Path
        Path to households spatial data file
        
    Returns
    -------
    gpd.GeoDataFrame
        Base firm table with region_sector, importance, and geometry
    """
    # Extract region_sectors
    firm_table = pd.DataFrame({
        'tuple': mrio.region_sectors,
        'region': [tup[0] for tup in mrio.region_sectors],
        'sector': [tup[1] for tup in mrio.region_sectors],
        'region_sector': mrio.region_sector_names
    })
    
    # Add importance
    tot_outputs_per_region_sector = mrio.sum(axis=1)
    firm_table['importance'] = firm_table['tuple'].map(tot_outputs_per_region_sector)
    check_successful_extraction(firm_table, "importance")
    
    # Add point (largest populated node)
    households_points = gpd.read_file(households_spatial)
    mapping_region_to_point = (households_points[households_points['region'].isin(mrio.regions)]
                               .groupby('region').apply(lambda df: df.loc[df['population'].idxmax(), "geometry"]))
    firm_table['geometry'] = firm_table['region'].map(mapping_region_to_point)
    firm_table = gpd.GeoDataFrame(firm_table, crs=households_points.crs)
    logging.info(f'Number of firms from MRIO only: {firm_table.shape}')
    
    return firm_table


def _integrate_disaggregated_data(firm_table: gpd.GeoDataFrame, firms_spatial: Path, mrio: Mrio) -> gpd.GeoDataFrame:
    """Integrate disaggregated spatial firm data if available.
    
    Parameters
    ----------
    firm_table : gpd.GeoDataFrame
        Base firm table from MRIO
    firms_spatial : Path
        Path to disaggregated firms spatial data
    mrio : Mrio
        Multi-regional input-output table
        
    Returns
    -------
    gpd.GeoDataFrame
        Firm table with disaggregated data integrated
    """
    disag_data = load_firms_spatial_data(firms_spatial, mrio.sectors, mrio.regions)
    
    if disag_data.empty:
        logging.info("No disaggregated data found")
        return firm_table
    
    logging.info("Processing disaggregated data")
    disag_firm_table = create_disag_firm_table(disag_data)
    
    # Remove the firms in the original table for which we have the disag info
    firm_table = firm_table[~firm_table['tuple'].isin(disag_firm_table['tuple'])].copy()
    
    # Add the disaggregated firms
    firm_table = pd.concat([firm_table, disag_firm_table])
    logging.info(f'Number of firms added from disaggregated data: {disag_firm_table.shape}')
    
    return firm_table


def _handle_internal_flows(firm_table: gpd.GeoDataFrame, mrio: Mrio, io_cutoff: float) -> gpd.GeoDataFrame:
    """Duplicate firms where internal region_sector flows exist.
    
    Parameters
    ----------
    firm_table : gpd.GeoDataFrame
        Firm table to process
    mrio : Mrio
        Multi-regional input-output table
    io_cutoff : float
        Input-output cutoff threshold
        
    Returns
    -------
    gpd.GeoDataFrame
        Firm table with duplicated firms for internal flows
    """
    logging.info("Duplicating firms where internal region_sector flows")
    
    # Get total outputs for importance assignment
    tot_outputs_per_region_sector = mrio.sum(axis=1)
    
    region_sectors_internal_flows = mrio.get_region_sectors_with_internal_flows(io_cutoff)
    region_sectors_one_firm = (firm_table['tuple'].value_counts()[firm_table['tuple'].value_counts() == 1]
                               .index.to_list())
    where_to_add_one_firm = list(set(region_sectors_internal_flows) & set(region_sectors_one_firm))
    
    if not where_to_add_one_firm:
        logging.info('No firms to duplicate for internal flows')
        return firm_table
    
    duplicated_firms = pd.DataFrame({
        'tuple': where_to_add_one_firm,
        'region': [tup[0] for tup in where_to_add_one_firm],
        'sector': [tup[1] for tup in where_to_add_one_firm],
        'region_sector': ['_'.join(tup) for tup in where_to_add_one_firm],
        'geometry': firm_table.set_index('tuple').loc[where_to_add_one_firm, 'geometry']
    })
    
    # Add importance
    duplicated_firms['importance'] = duplicated_firms['tuple'].map(tot_outputs_per_region_sector)
    check_successful_extraction(duplicated_firms, "importance")
    logging.info(f'Number of firms added for internal flows: {duplicated_firms.shape[0]}')
    
    # Merge with the firm table, and divide the importance by 2 where we added a firm
    firm_table = pd.concat([firm_table, duplicated_firms])
    firm_table.loc[firm_table['tuple'].isin(where_to_add_one_firm), 'importance'] = \
        firm_table.loc[firm_table['tuple'].isin(where_to_add_one_firm), 'importance'] / 2
    
    return firm_table


def _filter_small_firms(firm_table: gpd.GeoDataFrame, mrio: Mrio, cutoff_firm_output: dict, 
                       monetary_units_in_data: str) -> gpd.GeoDataFrame:
    """Filter out firms below output threshold while preserving top 2 per region_sector.
    
    Parameters
    ----------
    firm_table : gpd.GeoDataFrame
        Firm table to filter
    mrio : Mrio
        Multi-regional input-output table
    cutoff_firm_output : dict
        Output cutoff configuration
    monetary_units_in_data : str
        Monetary units used in data
        
    Returns
    -------
    gpd.GeoDataFrame
        Filtered firm table
    """
    # Add id
    firm_table['id'] = range(firm_table.shape[0])
    firm_table.index = firm_table['id']
    
    # Identify the top 2 firms per region sector
    top2_idx = (
        firm_table.groupby("region_sector")["importance"]
        .nlargest(2)
        .reset_index(level=0, drop=True)
        .index
    )
    top2_bool_index = firm_table.index.isin(top2_idx)
    
    # Calculate the estimated output per firm
    output_per_region_sector = mrio.get_total_output_per_region_sectors().to_dict()
    estimated_region_sector_output = firm_table['tuple'].map(output_per_region_sector)
    estimated_output = firm_table.groupby('region_sector', as_index=False, group_keys=False)['importance'] \
        .apply(lambda s: s / s.sum()).sort_index(ascending=True)
    estimated_output = estimated_output * estimated_region_sector_output
    
    cutoff = get_absolute_cutoff_value(cutoff_firm_output, monetary_units_in_data)
    logging.info(f'Filtering out firms with an estimated output '
                 f'of less than {cutoff} {monetary_units_in_data}')
    cond_low_output = estimated_output <= cutoff
    
    # Cut out those that are not in the top 2 and have estimated output below threshold
    firm_table = firm_table[(~cond_low_output) | top2_bool_index].copy()
    logging.info(f'Number of firms removed by the firm cutoff condition: {cond_low_output.sum()}')
    
    # Reset ids
    firm_table['id'] = range(firm_table.shape[0])
    firm_table.index = firm_table['id']
    
    # Add name
    firm_table['name'] = firm_table.groupby('region_sector').cumcount().astype(str)
    firm_table['name'] = firm_table['region_sector'] + '_' + firm_table['name']
    
    return firm_table


def _assign_transport_nodes(firm_table: gpd.GeoDataFrame, transport_nodes: gpd.GeoDataFrame, 
                           admin: Optional[list[str]]) -> pd.DataFrame:
    """Assign firms to nearest transport nodes and add coordinates.
    
    Parameters
    ----------
    firm_table : gpd.GeoDataFrame
        Firm table with geometry
    transport_nodes : gpd.GeoDataFrame
        Transport network nodes
    admin : Optional[list[str]]
        Administrative levels to include
        
    Returns
    -------
    pd.DataFrame
        Firm table with transport nodes and coordinates
    """
    logging.info("Assign firms to nearest road node")
    
    admissible_node_mode = ['roads', 'maritime']
    potential_nodes = transport_nodes[transport_nodes['type'].isin(admissible_node_mode)]
    firm_table['od_point'] = find_nearest_node_id(potential_nodes, firm_table)
    check_successful_extraction(firm_table, "od_point")
    
    if isinstance(admin, list):
        for admin_level in admin:
            firm_table[admin_level] = firm_table["od_point"].map(potential_nodes[admin_level])
    
    # Add long lat
    long_lat = get_long_lat(firm_table['od_point'], transport_nodes)
    firm_table['long'] = long_lat['long']
    firm_table['lat'] = long_lat['lat']
    
    return firm_table


def _enrich_sector_metadata(firm_table: pd.DataFrame, sector_table: pd.DataFrame, mrio: Mrio) -> pd.DataFrame:
    """Add sector type, usd_per_ton, margin, and transport_share.
    
    Parameters
    ----------
    firm_table : pd.DataFrame
        Firm table to enrich
    sector_table : pd.DataFrame
        Sector table
    mrio : Mrio
        Multi-regional input-output table
        
    Returns
    -------
    pd.DataFrame
        Enriched firm table with sector metadata
    """
    # Add sector type
    firm_table['sector_type'] = firm_table['region_sector'].map(sector_table.set_index('region_sector')['type'])
    check_successful_extraction(firm_table, "sector_type")

    # Add usd per ton
    firm_table['usd_per_ton'] = firm_table['region_sector'].map(sector_table.set_index('region_sector')['usd_per_ton'])

    # Add margin and transport input share
    present_industries = list(firm_table['tuple'].unique())
    firm_table['margin'] = firm_table['tuple'].map(mrio.get_margin_per_industry(present_industries))
    check_successful_extraction(firm_table, "margin")

    sector_to_type = firm_table[['sector', 'sector_type']].drop_duplicates().set_index('sector')[
        'sector_type'].to_dict()
    firm_table['transport_share'] = firm_table['tuple'].map(
        mrio.get_transport_input_share_per_industry(sector_to_type, present_industries))
    check_successful_extraction(firm_table, "transport_share")

    return firm_table


def define_firms_from_mrio(mrio: Mrio, sector_table: pd.DataFrame, households_spatial: Path, firms_spatial: Path,
                           transport_nodes: gpd.GeoDataFrame, io_cutoff: float, cutoff_firm_output: dict,
                           monetary_units_in_data: str, admin: Optional[list[str]] = None) -> pd.DataFrame:
    """Main orchestrator function for creating firms from MRIO data.
    
    This function coordinates the entire firm creation process by delegating
    to specialized helper functions, each handling a specific phase.
    
    Parameters
    ----------
    mrio : Mrio
        Multi-regional input-output table
    sector_table : pd.DataFrame
        Sector table
    households_spatial : Path
        Path to households spatial data
    firms_spatial : Path
        Path to disaggregated firms spatial data
    transport_nodes : gpd.GeoDataFrame
        Transport network nodes
    io_cutoff : float
        Input-output cutoff threshold
    cutoff_firm_output : dict
        Firm output cutoff configuration
    monetary_units_in_data : str
        Monetary units used in data
    admin : Optional[list[str]]
        Administrative levels to include
        
    Returns
    -------
    pd.DataFrame
        Complete firm table ready for firm creation
    """
    # 1. Create base firm table from MRIO
    firm_table = _create_base_firm_table_from_mrio(mrio, households_spatial)
    
    # 2. Integrate disaggregated spatial data
    firm_table = _integrate_disaggregated_data(firm_table, firms_spatial, mrio)
    
    # 3. Handle internal flows (duplicate firms where needed)
    firm_table = _handle_internal_flows(firm_table, mrio, io_cutoff)
    
    # 4. Filter out small firms based on output thresholds
    firm_table = _filter_small_firms(firm_table, mrio, cutoff_firm_output, monetary_units_in_data)
    
    # 5. Assign transport nodes and coordinates
    firm_table = _assign_transport_nodes(firm_table, transport_nodes, admin)
    
    # 6. Enrich with sector metadata
    firm_table = _enrich_sector_metadata(firm_table, sector_table, mrio)
    
    logging.info(f"Created {firm_table.shape[0]} firms in {firm_table['region'].nunique()} regions")
    return firm_table


def define_firms_from_network_data(
        filepath_firm_table: str,
        filepath_location_table: str,
        sectors_to_include: list[str],
        transport_nodes: pd.DataFrame,
        sector_table: pd.DataFrame
) -> pd.DataFrame:
    """Define firms based on the firm_table
    The output is a dataframe, 1 row = 1 firm.
    The instances Firms are created in the createFirm function.
    """
    # Load firm table
    firm_table = pd.read_csv(filepath_firm_table, dtype={'region': str})

    # Filter out some sectors
    if sectors_to_include != "all":
        firm_table = firm_table[firm_table['sector'].isin(sectors_to_include)]

    # Assign firms to the closest road nodes
    selected_regions = list(firm_table['region'].unique())
    logging.info('Select ' + str(firm_table.shape[0]) +
                 " firms in " + str(len(selected_regions)) + ' regions')
    location_table = gpd.read_file(filepath_location_table)
    cond_selected_regions = location_table['region'].isin(selected_regions)
    dic_region_to_points = location_table[cond_selected_regions].set_index('region')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_region_to_points.items()
    }
    firm_table['od_point'] = firm_table['region'].map(dic_region_to_road_node_id)

    # Information required by the createFirms function
    # add sector type
    sector_to_sector_type = sector_table.set_index('sector')['type']
    firm_table['sector_type'] = firm_table['sector'].map(sector_to_sector_type)
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table['long'] = firm_table['od_point'].map(road_node_id_to_long_lat['long'])
    firm_table['lat'] = firm_table['od_point'].map(road_node_id_to_long_lat['lat'])
    # add importance
    firm_table['importance'] = 10

    return firm_table


def load_mrio_tech_coefs(
        firms: Firms,
        mrio: Mrio,
        io_cutoff: float
) -> None:
    # Load tech coef
    region_sector_present = list(firms.get_properties('region_sector', output_type="set"))
    tech_coef_dict = mrio.get_tech_coef_dict(threshold=io_cutoff, selected_region_sectors=region_sector_present)

    # Inject into firms
    for firm in firms.values():
        if firm.region_sector in tech_coef_dict.keys():
            firm.input_mix = tech_coef_dict[firm.region_sector]
        else:
            firm.input_mix = {}

    logging.info('Technical coefficient loaded.')


def calibrate_input_mix(
        firms: Firms,
        firm_table: pd.DataFrame,
        sector_table: pd.DataFrame,
        filepath_transaction_table: str
) -> tuple[Firms, pd.DataFrame]:
    transaction_table = pd.read_csv(filepath_transaction_table)

    domestic_b2b_sales_per_firm = transaction_table.groupby('supplier_id')['transaction'].sum()
    firm_table['domestic_B2B_sales'] = firm_table['id'].map(domestic_b2b_sales_per_firm).fillna(0)
    firm_table['output'] = firm_table['domestic_B2B_sales'] + firm_table['final_demand'] + firm_table['exports']

    # Identify the sector of the products exchanged recorded in the transaction table and whether they are essential
    transaction_table['product_sector'] = transaction_table['supplier_id'].map(firm_table.set_index('id')['sector'])
    transaction_table['is_essential'] = transaction_table['product_sector'].map(
        sector_table.set_index('sector')['essential'])

    # Get input mix from this data
    def get_input_mix(transaction_from_unique_buyer: pd.DataFrame, firm_tab: pd.DataFrame) -> dict:
        output = firm_tab.set_index('id').loc[transaction_from_unique_buyer.name, 'output']
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
    for firm in firms.values():
        firm.input_mix = input_mix[firm.pid]

    return firms, transaction_table


def load_inventories(firms: Firms, inventory_duration_targets: dict, model_time_unit: str, sector_table: pd.DataFrame) -> None:
    """Load inventory duration target

    If inventory_duration_target is an integer, it is uniformly applied to all firms.
    If it its "inputed", then we use the targets defined in the file filepath_inventory_duration_targets. In that case,
    targets are sector specific, i.e., it varies according to the type of input and the sector of the buying firm.
    If both cases, we can add extra units of inventories:
    - uniformly, e.g., all firms have more inventories of all inputs,
    - to specific inputs, all firms have extra agricultural inputs,
    - to specific buying firms, e.g., all manufacturing firms have more of all inputs,
    - to a combination of both. e.g., all manufacturing firms have more of agricultural inputs.
    We can also add some noise on the distribution of inventories. Not yet implemented.

    Parameters
    ----------

    sector_table : pd.DataFrame
        Sector table
    model_time_unit
    firms : pandas.DataFrame
        the list of Firms generated from the createFirms function
    inventory_duration_targets
    """
    time_unit_in_days = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365
    }
    time_adjustment = time_unit_in_days[inventory_duration_targets['unit']] / time_unit_in_days[model_time_unit]

    if inventory_duration_targets['definition'] == "per_input_type":
        input_sector_to_type = sector_table.set_index('region_sector')['type']
        values = inventory_duration_targets['values']
        default = values['default']
        for firm in firms.values():
            firm.inventory_manager.inventory_duration_target = \
                {input_sector: max(1.0, time_adjustment * values.get(input_sector_to_type.get(input_sector), default))
                 for input_sector in firm.input_mix.keys()}

    elif inventory_duration_targets['definition'] == 'inputed':
        dic_sector_inventory = pd.read_csv(inventory_duration_targets['filepath']) \
            .set_index(['buying_sector', 'input_sector'])['inventory_duration_target'].to_dict()
        for firm in firms.values():
            firm.inventory_manager.inventory_duration_target = {
                input_sector: time_adjustment * max(inventory_duration_targets['minimum'],
                                                    dic_sector_inventory[(firm.sector, input_sector)])
                for input_sector in firm.input_mix.keys()
            }

    else:
        raise ValueError("Unknown value entered for 'inventory_duration_targets.definition'")
