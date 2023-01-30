from . import parameters_default
from . import parameters

if hasattr(parameters, "input_folder"):
    input_folder = parameters.input_folder
else:
    input_folder = parameters_default.input_folder

# Filepaths
import os
## Transport
filepaths = {
    # Tranort related data
    "transport_parameters": os.path.join('input', input_folder, 'Transport', 'transport_parameters.yaml'),
    "transport_modes": os.path.join('input', input_folder, 'Transport', 'transport_modes.csv'),
    "roads_nodes": os.path.join('input', input_folder, 'Transport', 'roads_nodes.geojson'),
    "roads_edges": os.path.join('input', input_folder, 'Transport', 'roads_edges.geojson'),
    "multimodal_edges": os.path.join('input', input_folder, 'Transport', 'multimodal_edges.geojson'),
    "maritime_nodes": os.path.join('input', input_folder, 'Transport', 'maritime_nodes.geojson'),
    "maritime_edges": os.path.join('input', input_folder, 'Transport', 'maritime_edges.geojson'),
    "airways_nodes": os.path.join('input', input_folder, 'Transport', 'airways_nodes.geojson'),
    "airways_edges": os.path.join('input', input_folder, 'Transport', 'airways_edges.geojson'),
    "waterways_nodes": os.path.join('input', input_folder, 'Transport', 'waterways_nodes.geojson'),
    "waterways_edges": os.path.join('input', input_folder, 'Transport', 'waterways_edges.geojson'),
    # National data
    "sector_table": os.path.join('input', input_folder, 'National', "59sector_2016_sector_table.csv"),
    "tech_coef": os.path.join('input', input_folder, 'National', "59sector_2016_tech_coef.csv"),
    "inventory_duration_targets": os.path.join('input', input_folder, 'National', "inventory_duration_targets.csv"),
    # District data
    "adminunit_data": os.path.join('input', input_folder, 'Subnational', "59sector_2015_canton_data.geojson"),
    # Trade
    "imports": os.path.join('input', input_folder, 'Trade', "59sector_2016_import_table.csv"),
    "exports": os.path.join('input', input_folder, 'Trade', "59sector_2016_export_table.csv"),
    "transit_matrix": os.path.join('input', input_folder, 'Trade', "2016_transit_matrix.csv"),
}
