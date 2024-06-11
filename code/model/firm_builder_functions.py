import warnings
import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd
import geopandas as gpd

from code.agents.firm import Firm, FirmList
from code.network.mrio import Mrio
from code.model.builder_functions import get_index_closest_point, get_closest_road_nodes, get_long_lat


def create_firms(
        firm_table: pd.DataFrame,
        keep_top_n_firms: object = None,
        reactivity_rate: float = 0.1,
        utilization_rate: float = 0.8
) -> FirmList:
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
    if "main_sector" not in firm_table.columns:  # case of MRIO
        firm_table['main_sector'] = firm_table['sector']
    # print(firm_table.head())
    # print(firm_table.iloc[0])
    firm_list = FirmList([
        Firm(i,
             sector=firm_table.loc[i, "sector"],
             sector_type=firm_table.loc[i, "sector_type"],
             main_sector=firm_table.loc[i, "main_sector"],
             odpoint=firm_table.loc[i, "od_point"],
             importance=firm_table.loc[i, 'importance'],
             name=firm_table.loc[i, 'name'],
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
    4. It combines the firms of the same sector that are in the same road node (case of 2 admin_unit close
    to the same road node)
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
        .groupby(['admin_unit', 'od_point', 'sector'], as_index=False) \
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
    # add name, not really useful
    firm_table_per_od_point['name'] = firm_table_per_od_point['od_point'].astype(str) + '-' + firm_table_per_od_point[
        'sector']
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


def define_firms_from_mrio_data(
        filepath_country_sector_table: str,
        filepath_region_table: str,
        transport_nodes: pd.DataFrame):
    # Adrien's function for Global

    # Load firm table
    firm_table = pd.read_csv(filepath_country_sector_table)

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
    print(firm_table)

    return firm_table


def check_successful_extraction(firm_table: pd.DataFrame, attribute: str):
    if firm_table[attribute].isnull().sum() > 0:
        logging.warning(f"Unsuccessful extraction of {attribute} for "
                        f"{firm_table.loc[firm_table[attribute].isnull(), 'name']}")


def define_firms_from_mrio(
        filepath_mrio: Path,
        filepath_sector_table: Path,
        filepath_region_table: Path,
        transport_nodes: gpd.GeoDataFrame,
        io_cutoff: float) -> pd.DataFrame:
    # Load mrio
    # mrio = pd.read_csv(filepath_mrio, index_col=0)
    mrio = Mrio.load_mrio_from_filepath(filepath_mrio)
    # Extract region_sectors
    # region_sectors = list(set(mrio.index) | set(mrio.columns))
    # region_sectors = [region_sector for region_sector in region_sectors if
    #                   len(region_sector) == 8]  # format 1011-FRE... :TODO a bit specific to Ecuador, change
    firm_table = pd.DataFrame({
        'tuple': mrio.region_sectors,
        'region': [tup[0] for tup in mrio.region_sectors],
        'main_sector': [tup[1] for tup in mrio.region_sectors],
        'sector': mrio.region_sector_names,
        'name': mrio.region_sector_names
    })
    region_sectors_internal_flows = mrio.get_region_sectors_with_internal_flows(io_cutoff)
    duplicated_firms = pd.DataFrame({
        'tuple': region_sectors_internal_flows,
        'region': [tup[0] for tup in region_sectors_internal_flows],
        'main_sector': [tup[1] for tup in region_sectors_internal_flows],
        'sector': ['_'.join(tup) for tup in region_sectors_internal_flows],
        'name': ['_'.join([tup[0], tup[1], "bis"]) for tup in region_sectors_internal_flows]
    })
    firm_table = pd.concat([firm_table, duplicated_firms])
    # region_sectors = ['_'.join(tup) for tup in mrio.columns]
    # # For region-sector with internal flows, need two firms
    # # region_sectors_internal_flows = list(set(mrio.index) & set(mrio.columns))
    # region_sectors_internal_flows = [tup for tup in mrio.columns if mrio.loc[tup, tup] > 0]
    # region_sectors = region_sectors + [region_sector + '-bis' for region_sector in region_sectors_internal_flows]
    # # Create firm_table
    # firm_table = pd.DataFrame({"name": region_sectors})
    # # firm_table['admin_unit'] = firm_table['name'].str.extract(r'([0-9]*)-[A-Z0-9]{3}')
    # firm_table['admin_unit'] = firm_table['name'].str.extract(r'([A-Z]{3})')
    # check_successful_extraction(firm_table, "admin_unit")
    # # firm_table['main_sector'] = firm_table['name'].str.extract(r'[0-9]*-([A-Z]{2}[A-Z0-9]{1})')
    # firm_table['main_sector'] = firm_table['name'].str[4:]
    # check_successful_extraction(firm_table, "main_sector")
    # firm_table['sector'] = firm_table['name'].str.replace('-bis', '')
    # check_successful_extraction(firm_table, "sector")
    logging.info(f"Select {firm_table.shape[0]} firms in {firm_table['region'].nunique()} regions")

    # Assign firms to the nearest road node
    firm_table['od_point'] = get_closest_road_nodes(firm_table['region'], transport_nodes, filepath_region_table)

    # Add long lat
    long_lat = get_long_lat(firm_table['od_point'], transport_nodes)
    firm_table['long'] = long_lat['long']
    firm_table['lat'] = long_lat['lat']

    # Add importance
    # row_intermediary = [row for row in mrio.index if len(row) == 8]
    tot_outputs_per_region_sector = mrio.sum(axis=1)
    firm_table['importance'] = firm_table['tuple'].map(tot_outputs_per_region_sector)
    firm_with_two_firms_same_region_sector = firm_table['tuple'].isin(region_sectors_internal_flows)
    firm_table.loc[firm_with_two_firms_same_region_sector, "importance"] = \
        firm_table.loc[firm_with_two_firms_same_region_sector, "importance"] / 2
    check_successful_extraction(firm_table, "importance")

    # Add sector type
    sector_table = pd.read_csv(filepath_sector_table)
    
    firm_table['sector_type'] = firm_table['sector'].map(sector_table.set_index('sector')['type'])
    check_successful_extraction(firm_table, "sector_type")

    # Add id (where is it created otherwise?)
    firm_table['id'] = range(firm_table.shape[0])

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


def load_technical_coefficients(
        firm_list: FirmList,
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


def load_mrio_tech_coefs(
        firm_list: FirmList,
        filepath_mrio: Path,
        io_cutoff: float
):
    # Load mrio
    mrio = Mrio.load_mrio_from_filepath(filepath_mrio)
    # mrio = pd.read_csv(filepath_mrio, header=[0, 1], index_col=[0, 1])  # TODO: class mrio
    # mrio = pd.read_csv(filepath_mrio, index_col=0)

    # Load technical coefficient matrix from data
    # region_sectors = [tup for tup in mrio.index if tup[1] != "Imports"]
    # tot_outputs = mrio.loc[region_sectors].sum(axis=1)
    # matrix_output = pd.concat([tot_outputs] * len(mrio.index), axis=1).transpose()
    # matrix_output.index = mrio.index
    # tech_coef_matrix = mrio[region_sectors] / matrix_output
    tech_coef_dict = mrio.get_tech_coef_dict(threshold=io_cutoff)

    # Load into firm_list
    for firm in firm_list:
        if firm.sector in tech_coef_dict.keys():
            firm.input_mix = tech_coef_dict[firm.sector]
        else:
            firm.input_mix = {}

    logging.info('Technical coefficient loaded.')


def calibrate_input_mix(
        firm_list: FirmList,
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
    for firm in firm_list:
        firm.input_mix = input_mix[firm.pid]

    return firm_list, transaction_table


def load_inventories(firm_list: list, inventory_duration_target: int | str,
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
    filepath_inventory_duration_targets
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
            raise ValueError("Unknown value given for 'inputs_with_extra_inventories' or "
                             "'buying_sectors_with_extra_inventories'. Should be a list of string or 'all'")

    if min_inventory > 0:
        for firm in firm_list:
            firm.inventory_duration_target = {
                input_sector: max(min_inventory, inventory)
                for input_sector, inventory in firm.inventory_duration_target.items()
            }

    logging.info('Inventory duration targets loaded')
    if extra_inventory_target:
        logging.info(f"Extra inventory duration: {extra_inventory_target} "
                     f"for inputs {inputs_with_extra_inventories} "
                     f"for buying sectors{buying_sectors_with_extra_inventories}")
