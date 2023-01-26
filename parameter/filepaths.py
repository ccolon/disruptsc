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
    ## Supply
    # "district_sector_importance": os.path.join('input', input_folder, 'Supply', 'district_sector_importance.csv'),
    "sector_table": os.path.join('input', input_folder, 'National', "4digit_sector_table.csv"),
    "tech_coef": os.path.join('input', input_folder, 'National', "2016_13sectors_tech_coef.csv"),
    "sector_cutoffs": os.path.join('input', input_folder, 'National', "13sectors_sector_firm_cutoffs.csv"),
    "inventory_duration_targets": os.path.join('input', input_folder, 'National', "inventory_duration_targets.csv"),
    # Canton data
    "adminunit_data": os.path.join('input', input_folder, 'Subnational', "canton_data.geojson"),
    ## Trade
    # "entry_points": os.path.join('input', input_folder, 'Trade', "entry_points.csv"),
    "imports": os.path.join('input', input_folder, 'Trade', "4digit_import_table.csv"),
    "exports": os.path.join('input', input_folder, 'Trade', "4digit_export_table.csv"),
    "transit_matrix": os.path.join('input', input_folder, 'Trade', "transit_matrix.csv"),
    ## Network
    "firm_table": os.path.join('input', input_folder, 'Network', "firm_table.csv"),
    "location_table": os.path.join('input', input_folder, 'Network', "location_table.geojson"),
    "transaction_table": os.path.join('input', input_folder, 'Network', "transaction_table.csv")
}
