import warnings
import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd
import geopandas as gpd

from disruptsc.agents.firm import Firm, Firms
from disruptsc.model.basic_functions import find_nearest_node_id
from disruptsc.network.mrio import Mrio
from disruptsc.model.builder_functions import get_index_closest_point, get_long_lat, get_absolute_cutoff_value


def create_firms(
        firm_table: pd.DataFrame,
        keep_top_n_firms: object = None,
        inventory_restoration_time: float = 4,
        utilization_rate: float = 0.8,
        capital_to_value_added_ratio: float = 4
) -> Firms:
    """Create the firms

    It uses firm_table from rescaleNbFirms

    Parameters
    ----------
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
        Firm(i,
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

    return firms


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
    # Create dic
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

    """
    # E. Add final demand per firm
    # evaluate share of population represented
    cond = region_eco_data['region'].isin(selected_regions)
    represented_pop = region_eco_data.loc[cond, 'population'].sum()
    total_population = region_eco_data['population'].sum()
    # evaluate final demand
    rel_pop = firm_table['population'] / total_population
    tot_demand_of_sector = firm_table['sector'].map(sector_table.set_index('sector')['final_demand'])
    firm_table['final_demand'] = rel_pop * tot_demand_of_sector
    # print info
    logging.info("{:.0f}%".format(represented_pop / total_population * 100)+
        " of population represented")
    logging.info("{:.0f}%".format(firm_table['final_demand'].sum() / sector_table['final_demand'].sum() * 100)+
        " of final demand is captured")
    logging.info("{:.0f}%".format(firm_table['final_demand'].sum() / \
        sector_table.set_index('sector').loc[sectors_to_include, 'final_demand'].sum() * 100)+
        " of final demand of selected sector is captured")
    """

    # F. Log information
    logging.info('Create ' + str(firm_table_per_od_point.shape[0]) + " firms in " +
                 str(firm_table_per_od_point['od_point'].nunique()) + ' od points')
    for sector, row in sector_table.set_index("sector").iterrows():
        if (sectors_to_include == "all") or (sector in sectors_to_include):
            if row["supply_data"] in region_eco_data.columns:
                cond = firm_table_per_od_point['sector'] == sector
                logging.info(f"Sector {sector}: create {cond.sum()} firms that covers " +
                             "{:.0f}%".format(firm_table_per_od_point.loc[cond, 'absolute_size'].sum()
                                              / region_eco_data[row['supply_data']].sum() * 100) +
                             f" of total {row['supply_data']}")
            else:
                cond = firm_table_per_od_point['sector'] == sector
                logging.info(f"Sector {sector}: since {row['supply_data']} is not in the data, "
                             f"create {cond.sum()} firms that covers " +
                             "{:.0f}%".format(firm_table_per_od_point.loc[cond, 'population'].sum()
                                              / region_eco_data["population"].sum() * 100) +
                             f" of population")

    return firm_table_per_od_point, firm_table_per_region


def define_firms_from_mrio_data(
        filepath_country_sector_table: str,
        filepath_region_table: str,
        transport_nodes: pd.DataFrame):
    # Adrien's function for Global

    # Load firm table
    firm_table = pd.read_csv(filepath_country_sector_table)

    # Duplicate the lines by concatenating the DataFrame with itself
    firm_table = pd.concat([firm_table] * 2, ignore_index=True)
    # Assign firms to the closest road nodes
    selected_regions = list(firm_table['country_ISO'].unique())
    logging.info('Select ' + str(firm_table.shape[0]) +
                 " firms in " + str(len(selected_regions)) + ' regions')
    location_table = gpd.read_file(filepath_region_table)
    cond_selected_regions = location_table['country_ISO'].isin(selected_regions)
    dic_region_to_points = location_table[cond_selected_regions].set_index('country_ISO')['geometry'].to_dict()
    road_nodes = transport_nodes[transport_nodes['type'] == "roads"]
    dic_region_to_road_node_id = {
        region: road_nodes.loc[get_index_closest_point(point, road_nodes), 'id']
        for region, point in dic_region_to_points.items()
    }
    firm_table['od_point'] = firm_table['country_ISO'].map(dic_region_to_road_node_id)

    # Information required by the createFirms function
    # add long lat
    od_point_table = road_nodes[road_nodes['id'].isin(firm_table['od_point'])].copy()
    od_point_table['long'] = od_point_table.geometry.x
    od_point_table['lat'] = od_point_table.geometry.y
    road_node_id_to_long_lat = od_point_table.set_index('id')[['long', 'lat']]
    firm_table['long'] = firm_table['od_point'].map(road_node_id_to_long_lat['long'])
    firm_table['lat'] = firm_table['od_point'].map(road_node_id_to_long_lat['lat'])
    # add importance
    firm_table['importance'] = 10

    # add id (where is it created otherwise?)
    firm_table = firm_table.reset_index().rename(columns={"index": "id"})

    return firm_table


def check_successful_extraction(firm_table: pd.DataFrame, attribute: str):
    if firm_table[attribute].isnull().sum() > 0:
        logging.warning(f"Unsuccessful extraction of {attribute} for "
                        f"{firm_table[firm_table[attribute].isnull()].index}")


def load_disag_data(folder_path: Path, accepted_sectors: list, accepted_regions: list) -> gpd.GeoDataFrame:
    """
    Load all geojson files within the folder defined in parameters, then check if they have a region column,
    if the geometries are Points, then concatenate them.

    Parameters
    ----------
    accepted_regions
    accepted_sectors
    folder_path : Path
        Path to the folder containing geojson files.

    Returns
    -------
    gpd.GeoDataFrame
        Concatenated GeoDataFrame containing all valid geojson files.
    """
    geojson_files = list(folder_path.glob("*.geojson"))
    logging.info(f'Processing files {geojson_files}')
    geo_dfs = []

    for file in geojson_files:
        gdf = gpd.read_file(file)
        if 'region' in gdf.columns and gdf.geometry.geom_type.eq('Point').all():
            useless_regions = list(set(gdf['region'].unique()) - set(accepted_regions))
            logging.info(f"In file {file}, the following regions will not be used: {useless_regions} because "
                         f"they are no part of the MRIO table: {accepted_regions}")
            gdf = gdf[~gdf['region'].isin(useless_regions)]
            useless_cols = [col for col in gdf.columns if col not in ['region', "geometry"] + accepted_sectors]
            logging.info(f"In file {file}, the following columns will not be used: {useless_cols} because "
                         f"they are no part of the list of sectors defined in the MRIO table: {accepted_sectors}")
            disag_sectors = list(set(gdf.columns) & set(accepted_sectors))
            gdf = gdf[['region', 'geometry'] + disag_sectors]
            geo_dfs.append(gdf)

    if geo_dfs:
        concatenated_gdf = gpd.GeoDataFrame(pd.concat(geo_dfs, ignore_index=True))
        return concatenated_gdf
    else:
        return gpd.GeoDataFrame()


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


def define_firms_from_mrio(mrio: Mrio, filepath_sectors: Path, filepath_regions: Path, path_disag: Path,
                           transport_nodes: gpd.GeoDataFrame, io_cutoff: float, cutoff_firm_output: dict,
                           monetary_units_in_data: str) -> pd.DataFrame:
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
    region_table = gpd.read_file(filepath_regions)
    mapping_region_to_point = (region_table[region_table['region'].isin(mrio.regions)]
                               .groupby('region').apply(lambda df: df.loc[df['population'].idxmax(), "geometry"]))
    firm_table['geometry'] = firm_table['region'].map(mapping_region_to_point)
    firm_table = gpd.GeoDataFrame(firm_table, crs=region_table.crs)
    logging.info(f'Number of firms from MRIO only: {firm_table.shape}')

    # For some region sector, we have further data
    disag_data = load_disag_data(path_disag, mrio.sectors, mrio.regions)
    # disag_data = pd.DataFrame()
    if disag_data.empty:
        logging.info("No disaggregated data found")
    else:
        logging.info("Processing disaggregated data")
        disag_firm_table = create_disag_firm_table(disag_data)
        # disag_firm_table = disag_firm_table.iloc[:100]
        # Remove the firms in the original table for which we have the disag info
        firm_table = firm_table[~firm_table['tuple'].isin(disag_firm_table['tuple'])]
        # Add the disaggregated firms
        firm_table = pd.concat([firm_table, disag_firm_table])
        logging.info(f'Number of firms added from disaggregated data: {disag_firm_table.shape}')

    # For each region_sector, in which there are internal flows, if there are only one firms, add another one
    logging.info("Duplicating firms where internal region_sector flows")
    region_sectors_internal_flows = mrio.get_region_sectors_with_internal_flows(io_cutoff)
    region_sectors_one_firm = (firm_table['tuple'].value_counts()[firm_table['tuple'].value_counts() == 1]
                               .index.to_list())
    where_to_add_one_firm = list(set(region_sectors_internal_flows) & set(region_sectors_one_firm))
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

    if duplicated_firms.shape[0] > 0:
        # Merge with the firm table, and divide the importance by 2 where we added a firm
        firm_table = pd.concat([firm_table, duplicated_firms])
        firm_table.loc[firm_table['tuple'].isin(where_to_add_one_firm), 'importance'] = \
            firm_table.loc[firm_table['tuple'].isin(where_to_add_one_firm), 'importance'] / 2

    # Add id
    firm_table['id'] = range(firm_table.shape[0])
    firm_table.index = firm_table['id']

    # Filter out too small firms (note that, if one firm got duplicated and then got filtered out
    # both duplicates will be filtered out, so no consequence for internal flows)
    output_per_region_sector = mrio.get_total_output_per_region_sectors().to_dict()
    estimated_region_sector_output = firm_table['tuple'].map(output_per_region_sector)
    estimated_output = firm_table.groupby('region_sector', as_index=False, group_keys=False)['importance']\
        .apply(lambda s: s / s.sum()).sort_index(ascending=True)
    estimated_output = estimated_output * estimated_region_sector_output
    cutoff = get_absolute_cutoff_value(cutoff_firm_output, monetary_units_in_data)
    logging.info(f'Filtering out firms with an estimated output '
                 f'of less than {cutoff} {monetary_units_in_data}')
    cond_low_output = estimated_output <= cutoff
    firm_table = firm_table[~cond_low_output]
    logging.info(f'Number of firms removed by the firm cutoff condition: {cond_low_output.sum()}')

    # Reset ids
    firm_table['id'] = range(firm_table.shape[0])
    firm_table.index = firm_table['id']

    # Add name
    firm_table['name'] = firm_table.groupby('region_sector').cumcount().astype(str)
    firm_table['name'] = firm_table['region_sector'] + '_' + firm_table['name']

    # Assign firms to the nearest road node
    logging.info("Assign firms to nearest road node")
    # unique_points_with_firms = list(firm_table['geometry'].unique())
    # corresponding_od_point_ids = {point.wkt: get_index_closest_point(point, transport_nodes)
    #                               for point in unique_points_with_firms}
    # firm_table['od_point'] = firm_table['geometry'].map(lambda point: corresponding_od_point_ids[point.wkt])
    firm_table['od_point'] = find_nearest_node_id(transport_nodes, firm_table)
    check_successful_extraction(firm_table, "od_point")

    # Add long lat
    long_lat = get_long_lat(firm_table['od_point'], transport_nodes)
    firm_table['long'] = long_lat['long']
    firm_table['lat'] = long_lat['lat']

    # Add sector type
    sector_table = pd.read_csv(filepath_sectors)  # should use sector only...
    firm_table['sector_type'] = firm_table['region_sector'].map(sector_table.set_index('sector')['type'])
    check_successful_extraction(firm_table, "sector_type")

    # Add usd per ton
    firm_table['usd_per_ton'] = firm_table['region_sector'].map(sector_table.set_index('sector')['usd_per_ton'])

    logging.info(f"Create {firm_table.shape[0]} firms in {firm_table['region'].nunique()} regions")

    return firm_table


def define_firms_from_network_data(
        filepath_firm_table: str,
        filepath_location_table: str,
        sectors_to_include: list,
        transport_nodes: pd.DataFrame,
        filepath_sector_table: str
):
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
    sector_table = pd.read_csv(filepath_sector_table)
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


def load_technical_coefficients(
        firms: Firms,
        filepath_tech_coef: str,
        io_cutoff: float = 0.1,
        import_sector_name: str | None = "IMP"
):
    """Load the input mix of the firms' Leontief function

    Parameters
    ----------
    firms :
        the list of Firms generated from the createFirms function

    filepath_tech_coef : string
        Filepath to the matrix of technical coefficients

    io_cutoff : float
        Filters out technical coefficient below this cutoff value. Default to 0.1.

    import_sector_name : None or string
        Give the name of the import sector. If None, then the import technical coefficient is discarded. Default to None

    Returns
    -------
    list of Firms
    """

    # Load technical coefficient matrix from data
    tech_coef_matrix = pd.read_csv(filepath_tech_coef, index_col=0)
    tech_coef_matrix = tech_coef_matrix.mask(tech_coef_matrix <= io_cutoff, 0)

    # We select only the technical coefficient between sectors that are actually represented in the economy
    # Note that, when filtering out small sector-district combination, some sector may not be present.
    region_sector_present = list(firms.get_properties('region_sector', output_type="set"))
    if import_sector_name:
        tech_coef_matrix = tech_coef_matrix.loc[region_sector_present + [import_sector_name], region_sector_present]
    else:
        tech_coef_matrix = tech_coef_matrix.loc[region_sector_present, region_sector_present]

    # Check whether all sectors have input
    cond_sector_no_inputs = tech_coef_matrix.sum() == 0
    if cond_sector_no_inputs.any():
        warnings.warn(
            'Some sectors have no inputs: ' + str(cond_sector_no_inputs[cond_sector_no_inputs].index.to_list())
            + " Check this sector or reduce the io_coef cutoff")

    # Load input mix
    for firm in firms.values():
        firm.input_mix = tech_coef_matrix.loc[tech_coef_matrix.loc[:, firm.sector] != 0, firm.sector].to_dict()

    logging.info('Technical coefficient loaded. io_cutoff: ' + str(io_cutoff))


def load_mrio_tech_coefs(
        firms: Firms,
        mrio: Mrio,
        io_cutoff: float,
        monetary_units_in_data: str
):
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
):
    transaction_table = pd.read_csv(filepath_transaction_table)

    domestic_b2b_sales_per_firm = transaction_table.groupby('supplier_id')['transaction'].sum()
    firm_table['domestic_B2B_sales'] = firm_table['id'].map(domestic_b2b_sales_per_firm).fillna(0)
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
    for firm in firms.values():
        firm.input_mix = input_mix[firm.pid]

    return firms, transaction_table


def load_inventories(firms: Firms, inventory_duration_target: int | str, given_time_unit: str, model_time_unit: str,
                     filepath_inventory_duration_targets: Path, extra_inventory_target: int | None = None,
                     inputs_with_extra_inventories: None | list = None,
                     buying_sectors_with_extra_inventories: None | list = None,
                     min_inventory: int = 1):
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
    model_time_unit
    given_time_unit
    filepath_inventory_duration_targets
    firms : pandas.DataFrame
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
    """
    time_unit_in_days = {
        "day": 1,
        "week": 7,
        "month": 30,
        "year": 365
    }
    time_adjustment = time_unit_in_days[given_time_unit] / time_unit_in_days[model_time_unit]

    if isinstance(inventory_duration_target, int):
        for firm in firms.values():
            firm.inventory_duration_target = {input_sector: time_adjustment * inventory_duration_target
                                              for input_sector in firm.input_mix.keys()}

    elif inventory_duration_target == 'inputed':
        dic_sector_inventory = pd.read_csv(filepath_inventory_duration_targets) \
            .set_index(['buying_sector', 'input_sector'])['inventory_duration_target'].to_dict()
        for firm in firms.values():
            firm.inventory_duration_target = {
                input_sector: time_adjustment * dic_sector_inventory[(firm.sector, input_sector)]
                for input_sector in firm.input_mix.keys()
            }

    else:
        raise ValueError("Unknown value entered for 'inventory_duration_target'")

    # if random_mean_sd:
    #     if random_draw:
    #         for firm in firms:
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
            for firm in firms.values():
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[
                                      input_sector] + extra_inventory_target * time_adjustment
                    if (input_sector in inputs_with_extra_inventories) else firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif (inputs_with_extra_inventories == 'all') and isinstance(buying_sectors_with_extra_inventories, list):
            for firm in firms.values():
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[
                                      input_sector] + extra_inventory_target * time_adjustment
                    if (firm.sector in buying_sectors_with_extra_inventories) else firm.inventory_duration_target[
                        input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif isinstance(inputs_with_extra_inventories, list) and isinstance(buying_sectors_with_extra_inventories,
                                                                            list):
            for firm in firms.values():
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[
                                      input_sector] + extra_inventory_target * time_adjustment
                    if ((input_sector in inputs_with_extra_inventories) and (
                            firm.sector in buying_sectors_with_extra_inventories)) else
                    firm.inventory_duration_target[input_sector]
                    for input_sector in firm.input_mix.keys()
                }

        elif (inputs_with_extra_inventories == 'all') and (buying_sectors_with_extra_inventories == 'all'):
            for firm in firms.values():
                firm.inventory_duration_target = {
                    input_sector: firm.inventory_duration_target[
                                      input_sector] + extra_inventory_target * time_adjustment
                    for input_sector in firm.input_mix.keys()
                }

        else:
            raise ValueError("Unknown value given for 'inputs_with_extra_inventories' or "
                             "'buying_sectors_with_extra_inventories'. Should be a list of string or 'all'")

    if min_inventory > 0:
        for firm in firms.values():
            firm.inventory_duration_target = {
                input_sector: max(min_inventory * time_adjustment, inventory)
                for input_sector, inventory in firm.inventory_duration_target.items()
            }

    logging.info('Inventory duration targets loaded')
    if extra_inventory_target:
        logging.info(f"Extra inventory duration: {extra_inventory_target} "
                     f"for inputs:  {inputs_with_extra_inventories} "
                     f"for buying sectors: {buying_sectors_with_extra_inventories}")
