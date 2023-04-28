from old_code.parameter import parameters_default, parameters
import os
from pathlib import Path

project_directory = Path(os.path.abspath(__file__)).parents[1]

if hasattr(parameters, "input_folder"):
    input_folder = parameters.input_folder
else:
    input_folder = parameters_default.input_folder


def create_path(data_type, filename):
    return os.path.join(project_directory, 'input', input_folder, data_type, filename)


filepaths = {
    # Transport related data
    "transport_parameters": create_path('Transport', 'transport_parameters.yaml'),
    "transport_modes": create_path('Transport', 'transport_modes.csv'),
    "roads_nodes": create_path('Transport', 'roads_nodes.geojson'),
    "roads_edges": create_path('Transport', 'roads_edges.geojson'),
    "multimodal_edges": create_path('Transport', 'multimodal_edges.geojson'),
    "maritime_nodes": create_path('Transport', 'maritime_nodes.geojson'),
    "maritime_edges": create_path('Transport', 'maritime_edges.geojson'),
    "airways_nodes": create_path('Transport', 'airways_nodes.geojson'),
    "airways_edges": create_path('Transport', 'airways_edges.geojson'),
    "waterways_nodes": create_path('Transport', 'waterways_nodes.geojson'),
    "waterways_edges": create_path('Transport', 'waterways_edges.geojson'),
    # National data
    "sector_table": create_path('National', "59sector_2016_sector_table.csv"),
    "tech_coef": create_path('National', "59sector_2016_tech_coef.csv"),
    "inventory_duration_targets": create_path('National', "59sector_inventory_targets.csv"),
    # District data
    "adminunit_data": create_path('Subnational', "59sector_2015_canton_data.geojson"),
    # Trade
    "imports": create_path('Trade', "59sector_2016_import_table.csv"),
    "exports": create_path('Trade', "59sector_2016_export_table.csv"),
    "transit_matrix": create_path('Trade', "2016_transit_matrix.csv"),
    ## Network
    "firm_table": create_path('Network', "firm_table.csv"),
    "location_table": create_path('Network', "location_table.geojson"),
    "transaction_table": create_path('Network', "transaction_table.csv")
}
