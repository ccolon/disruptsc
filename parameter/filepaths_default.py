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
    "roads_edges": os.path.join('input', input_folder, 'Transport', 'roads_edges.geojson'),
    "railways_nodes": os.path.join('input', input_folder, 'Transport', 'railways_nodes.geojson'),
    "railways_edges": os.path.join('input', input_folder, 'Transport', 'railways_edges.geojson'),
    "airways_nodes": os.path.join('input', input_folder, 'Transport', 'airways_nodes.geojson'),
    "airways_edges": os.path.join('input', input_folder, 'Transport', 'airways_edges.geojson'),
    "waterways_nodes": os.path.join('input', input_folder, 'Transport', 'waterways_nodes.geojson'),
    "waterways_edges": os.path.join('input', input_folder, 'Transport', 'waterways_edges.geojson'),
    "multimodal_edges": os.path.join('input', input_folder, 'Transport', 'multimodal_edges.geojson'),
    "maritime_nodes": os.path.join('input', input_folder, 'Transport', 'maritime_nodes.geojson'),
    "maritime_edges": os.path.join('input', input_folder, 'Transport', 'maritime_edges.geojson'),
    "extra_roads_edges": os.path.join('input', input_folder, 'Transport', "road_edges_extra.geojson"),
    ## National
    "sector_table": os.path.join('input', input_folder, 'National', "sector_table.csv"),
    "tech_coef": os.path.join('input', input_folder, 'National', "tech_coef_matrix.csv"),
    "inventory_duration_targets": os.path.join('input', input_folder, 'National', "inventory_duration_targets.csv"),
    "sector_cutoffs": os.path.join('input', input_folder, 'National', "sector_firm_cutoffs.csv"),
    # Subnational
    "adminunit_economic_data": os.path.join('input', input_folder, 'Subnational', "district_economic_data.geojson"),
    ## Trade
    # "entry_points": os.path.join('input', input_folder, 'Trade', "entry_points.csv"),
    "imports": os.path.join('input', input_folder, 'Trade', "import_table.csv"),
    "exports": os.path.join('input', input_folder, 'Trade', "export_table.csv"),
    "transit_matrix": os.path.join('input', input_folder, 'Trade', "transit_matrix.csv"),
}
