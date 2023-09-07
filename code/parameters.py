import importlib
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from dataclasses import dataclass

from code import paths


@dataclass
class Parameters:
    region: str
    export_details: dict
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
    min_nb_firms_per_sector: int
    local_demand_cutoff: float
    countries_to_include: str | list
    logistic_modes: str
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
    force_local_retailer: bool
    disruption_description: dict
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
    transport_cost_noise_level: float
    firm_sampling_mode: str
    filepaths: dict
    export_files: bool
    simulation_type: str
    transport_cost_data: dict
    export_folder: Path | str = ""

    @classmethod
    def load_default_parameters(cls, parameter_folder: Path):
        with open(parameter_folder / "default.yaml", 'r') as f:
            default_parameters = yaml.safe_load(f)
        return cls(**default_parameters)

    @classmethod
    def load_parameters(cls, parameter_folder: Path, region: str):
        # Load default and user_defined parameters
        with open(parameter_folder / "default.yaml", 'r') as f:
            parameters = yaml.safe_load(f)
        user_defined_parameter_filepath = parameter_folder / f"user_defined_{region}.yaml"
        if os.path.exists(user_defined_parameter_filepath):
            logging.info(f'User defined parameter file found for {region}')
            with open(parameter_folder / f"user_defined_{region}.yaml", 'r') as f:
                overriding_parameters = yaml.safe_load(f)
            # Merge both
            for key, val in parameters.items():
                if key in overriding_parameters:
                    if isinstance(val, dict):
                        cls.merge_dict_with_priority(parameters[key], overriding_parameters[key])
                    else:
                        parameters[key] = overriding_parameters[key]
        else:
            logging.info(f'No user defined parameter file found named user_defined_{region}.yaml, '
                         f'using default parameters')
        # Load region
        parameters['region'] = region
        # Create parameters
        parameters = cls(**parameters)
        # Adjust filepath
        parameters.build_full_filepath()
        # Create export folder

        # Cast datatype
        parameters.epsilon_stop_condition = float(parameters.epsilon_stop_condition)
        parameters.duration_dic = {int(key): val for key, val in parameters.duration_dic.items()}

        return parameters

    @staticmethod
    def merge_dict_with_priority(default_dict: dict, overriding_dict: dict):
        for key, val in default_dict.items():
            if key in overriding_dict:
                default_dict[key] = overriding_dict[key]

    def build_full_filepath(self):
        for key, val in self.filepaths.items():
            self.filepaths[key] = paths.INPUT_FOLDER / self.region / val

    def export(self):
        with open(self.export_folder / 'parameters.yaml', 'w') as file:
            yaml.dump(self, file)

    def create_export_folder(self):
        if not os.path.isdir(paths.OUTPUT_FOLDER / self.region):
            os.mkdir(paths.OUTPUT_FOLDER / self.region)
        self.export_folder = paths.OUTPUT_FOLDER / self.region / datetime.now().strftime('%Y%m%d_%H%M%S')
        os.mkdir(self.export_folder)

    def adjust_logging_behavior(self):
        if self.logging_level == "info":
            logging_level = logging.INFO
        else:
            logging_level = logging.DEBUG

        if self.export_files:
            importlib.reload(logging)
            logging.basicConfig(
                filename=self.export_folder / 'exp.log',
                level=logging_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logging.getLogger().addHandler(logging.StreamHandler())
        else:
            importlib.reload(logging)
            logging.basicConfig(
                level=logging_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
