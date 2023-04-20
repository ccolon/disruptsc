from pathlib import Path

import yaml
from dataclasses import dataclass

from code import paths


@dataclass
class Parameters:
    input_folder: str
    export: dict
    specific_edges_to_monitor: dict
    logging_level: str
    transport_modes: list
    monetary_units_in_model: str
    monetary_units_inputed: str
    firm_data_type: str
    congestion: bool
    propagate_input_price_change: bool
    sectors_to_include: str
    sectors_to_exclude: list | None
    sectors_no_transport_network: list
    cutoff_sector_output: dict
    cutoff_sector_demand: dict
    combine_sector_cutoff: str
    districts_to_include: str | list
    pop_density_cutoff: float
    pop_cutoff: float
    local_demand_cutoff: float
    countries_to_include: str | list
    logistics_modes: str
    district_sector_cutoff: str
    nb_top_district_per_sector: None | int
    explicit_service_firm: bool
    inventory_duration_target: str | int
    extra_inventory_target: None | int
    inputs_with_extra_inventories: str | list
    buying_sectors_with_extra_inventories: str | list
    reactivity_rate: float
    utilization_rate: float
    io_cutoff: float
    rationing_mode: str
    nb_suppliers_per_input: float
    weight_localization_firm: float
    weight_localization_household: float
    disruption_analysis: None | dict
    time_resolution: str
    nodeedge_tested_topn: None | int
    nodeedge_tested_skipn: None | int
    model_IO: bool
    duration_dic: dict
    extra_roads: bool
    epsilon_stop_condition: float
    route_optimization_weight: str
    cost_repercussion_mode: str
    account_capacity: bool
    firm_sampling_mode: str
    filepaths: dict

    @classmethod
    def load_default_parameters(cls, parameter_folder: Path):
        with open(parameter_folder / "default.yaml", 'r') as f:
            default_parameters = yaml.safe_load(f)
        return cls(**default_parameters)

    @classmethod
    def load_parameters(cls, parameter_folder: Path):
        # Load default and user_defined parameters
        with open(parameter_folder / "default.yaml", 'r') as f:
            parameters = yaml.safe_load(f)
        with open(parameter_folder / "user_defined.yaml", 'r') as f:
            overriding_parameters = yaml.safe_load(f)
        # Merge both
        for key, val in parameters.items():
            if key in overriding_parameters:
                parameters[key] = overriding_parameters[key]
        # Adjust path
        parameters = cls(**parameters)
        # Adjust filepath
        parameters.build_full_filepath()

        return parameters

    def build_full_filepath(self):
        for key, val in self.filepaths.items():
            self.filepaths[key] = paths.INPUT_FOLDER / self.input_folder / val
