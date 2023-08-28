import logging
from pathlib import Path

import geopandas
import pandas
import pandas as pd

from code.agents.country import Country, CountryList
from code.model.functions import rescale_monetary_values


def create_countries(filepath_imports: Path, filepath_exports: Path, filepath_transit: Path,
                     transport_nodes: geopandas.GeoDataFrame, present_sectors: list,
                     countries_to_include: list | str = 'all', time_resolution: str = "week",
                     target_units: str = "mUSD", input_units: str = "USD") -> CountryList:
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
        cond_country = transport_nodes['special'] == country
        od_point = transport_nodes.loc[cond_country, "id"]
        lon = transport_nodes.geometry.x
        lat = transport_nodes.geometry.y
        if len(od_point) == 0:
            raise ValueError('No od_point found for ' + country)
        elif len(od_point) > 2:
            raise ValueError('More than 1 od_point for ' + country)
        else:
            od_point = od_point.iloc[0]
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
        # Note that transit are not given per sector, so, if we only consider a few sector,
        # the full transit flows will still be used
        transit_from = transit_matrix.loc[:, country]
        transit_from = transit_from[transit_from > 0].to_dict()
        transit_to = transit_matrix.loc[country, :]
        transit_to = transit_to[transit_to > 0].to_dict()

        # create the list of Country object
        country_list += [Country(pid=country,
                                 qty_sold=qty_sold,
                                 qty_purchased=qty_purchased,
                                 odpoint=od_point,
                                 long=lon,
                                 lat=lat,
                                 transit_from=transit_from,
                                 transit_to=transit_to,
                                 supply_importance=supply_importance
                                 )]
    country_list = CountryList(country_list)

    logging.info('Country_list created: ' + str([country.pid for country in country_list]))

    return country_list


def create_countries_from_mrio(filepath_mrio: Path,
                               transport_nodes: geopandas.GeoDataFrame, time_resolution: str,
                               target_units: str, input_units: str) -> CountryList:
    logging.info('Creating country_list.')

    # Load mrio
    mrio = pd.read_csv(filepath_mrio, index_col=0)
    # Extract countries from MRIO
    buying_countries = [col for col in mrio.columns if len(col) == 3]  # TODO a bit specific to Ecuador, change
    selling_countries = [col for col in mrio.index if len(col) == 3]
    countries = list(set(buying_countries) | set(selling_countries))

    # Create country table
    country_table = pd.DataFrame({"pid": countries})
    logging.info(f"Select {country_table.shape[0]} countries")

    # Extract import, export, and transit matrices
    importing_region_sectors = [col for col in mrio.columns if len(col) == 8]
    import_table = rescale_monetary_values(
        mrio.loc[selling_countries, importing_region_sectors],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    exporting_region_sectors = [row for row in mrio.index if len(row) == 8]
    export_table = rescale_monetary_values(
        mrio.loc[exporting_region_sectors, buying_countries],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )
    transit_matrix = rescale_monetary_values(
        mrio.loc[selling_countries, buying_countries],
        time_resolution=time_resolution,
        target_units=target_units,
        input_units=input_units
    )

    logging.info("Total imports per " + time_resolution + " is " +
                 "{:.01f} ".format(import_table.sum().sum()) + target_units)
    logging.info("Total exports per " + time_resolution + " is " +
                 "{:.01f} ".format(export_table.sum().sum()) + target_units)
    logging.info("Total transit per " + time_resolution + " is " +
                 "{:.01f} ".format(transit_matrix.sum().sum()) + target_units)

    country_list = []
    total_imports = import_table.sum().sum()
    for country in countries:
        cond_country = transport_nodes['special'] == country
        od_point = transport_nodes.loc[cond_country, "id"]
        lon = transport_nodes.geometry.x
        lat = transport_nodes.geometry.y
        if len(od_point) == 0:
            raise ValueError('No odpoint found for ' + country)
        elif len(od_point) > 2:
            raise ValueError('More than 1 odpoint for ' + country)
        else:
            od_point = od_point.iloc[0]
            lon = lon.iloc[0]
            lat = lat.iloc[0]

        # imports, i.e., sales of countries
        if country in selling_countries:
            qty_sold = import_table.loc[country, :]
            qty_sold = qty_sold[qty_sold > 0].to_dict()
            supply_importance = sum(qty_sold.values()) / total_imports
        else:
            qty_sold = {}
            supply_importance = 0

        # exports, i.e., purchases from countries
        if country in buying_countries:
            qty_purchased = export_table.loc[:, country]
            qty_purchased = qty_purchased[qty_purchased > 0].to_dict()
        else:
            qty_purchased = {}

        # transits
        # Note that transit are not given per sector, so, if we only consider a few sector,
        # the full transit flows will still be used
        if country in transit_matrix.columns:
            transit_from = transit_matrix.loc[:, country]
            transit_from = transit_from[transit_from > 0].to_dict()
        else:
            transit_from = {}
        if country in transit_matrix.index:
            transit_to = transit_matrix.loc[country, :]
            transit_to = transit_to[transit_to > 0].to_dict()
        else:
            transit_to = {}

        # create the list of Country object
        country_list += [Country(pid=country,
                                 qty_sold=qty_sold,
                                 qty_purchased=qty_purchased,
                                 odpoint=od_point,
                                 long=lon,
                                 lat=lat,
                                 transit_from=transit_from,
                                 transit_to=transit_to,
                                 supply_importance=supply_importance
                                 )]
    country_list = CountryList(country_list)

    logging.info('Country_list created: ' + str([country.pid for country in country_list]))

    return country_list


