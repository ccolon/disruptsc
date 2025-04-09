import importlib
import logging
import os
from datetime import datetime
from pathlib import Path

import yaml
from dataclasses import dataclass

from disruptsc import paths
from disruptsc.model.basic_functions import rescale_monetary_values, draw_lognormal_samples

EPSILON = 1e-6
TRANSPORT_MALUS = rescale_monetary_values(1e9, "USD", "week", "USD", "week")
import_code = "IMP"


@dataclass
class Parameters:
    scope: str
    export_details: dict
    # specific_edges_to_monitor: dict
    t_final: int
    flow_data: dict
    parameters_to_calibrate: dict
    logging_level: str
    with_transport: bool
    transport_modes: list
    monetary_units_in_model: str
    monetary_units_in_data: str
    firm_data_type: str
    congestion: bool
    propagate_input_price_change: bool
    sectors_to_include: str
    sectors_to_exclude: list | None
    sectors_no_transport_network: list
    transport_to_households: bool
    cutoff_sector_output: dict
    cutoff_sector_demand: dict
    combine_sector_cutoff: str
    cutoff_firm_output: dict
    cutoff_household_demand: dict
    districts_to_include: str | list
    pop_density_cutoff: float
    pop_cutoff: float
    min_nb_firms_per_sector: int
    local_demand_cutoff: float
    countries_to_include: str | list
    district_sector_cutoff: str
    nb_top_district_per_sector: None | int
    explicit_service_firm: bool
    inventory_duration_targets: dict
    inventory_restoration_time: float
    utilization_rate: float
    io_cutoff: float
    rationing_mode: str
    nb_suppliers_per_input: float
    weight_localization_firm: float
    weight_localization_household: float
    force_local_retailer: bool
    events: list
    criticality: None | dict
    time_resolution: str
    nodeedge_tested_topn: None | int
    nodeedge_tested_skipn: None | int
    model_IO: bool
    duration_dic: dict
    extra_roads: bool
    epsilon_stop_condition: float
    route_optimization_weight: str
    cost_repercussion_mode: str
    price_increase_threshold: float
    capacity_constraint: bool
    transport_cost_noise_level: float
    firm_sampling_mode: str
    filepaths: dict
    export_files: bool
    simulation_type: str
    adaptive_inventories: bool
    adaptive_supplier_weight: bool
    transport_cost_data: dict
    capital_to_value_added_ratio: float
    logistics: dict
    mc_repetitions: int
    export_folder: Path | str = ""

    @classmethod
    def load_default_parameters(cls, parameter_folder: Path):
        with open(parameter_folder / "default.yaml", 'r') as f:
            default_parameters = yaml.safe_load(f)
        return cls(**default_parameters)

    @classmethod
    def load_parameters(cls, parameter_folder: Path, scope: str):
        # Load default and user_defined parameters
        with open(parameter_folder / "default.yaml", 'r') as f:
            parameters = yaml.safe_load(f)
        user_defined_parameter_filepath = parameter_folder / f"user_defined_{scope}.yaml"
        if os.path.exists(user_defined_parameter_filepath):
            logging.info(f'User defined parameter file found for {scope}')
            with open(parameter_folder / f"user_defined_{scope}.yaml", 'r') as f:
                overriding_parameters = yaml.safe_load(f)
            # Merge both
            for key, val in parameters.items():
                if key in overriding_parameters:
                    if isinstance(val, dict):
                        cls.merge_dict_with_priority(parameters[key], overriding_parameters[key])
                    else:
                        parameters[key] = overriding_parameters[key]
        else:
            logging.info(f'No user defined parameter file found named user_defined_{scope}.yaml, '
                         f'using default parameters')
        # Load scope
        parameters['scope'] = scope
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
        for key, val in overriding_dict.items():
            default_dict[key] = val
            # if key in default_dict:
            #     default_dict[key] = overriding_dict[key]
            # else:
            #     default_dict[key] = val

    def get_full_filepath(self, filepath):
        return paths.INPUT_FOLDER / self.scope / filepath

    def build_full_filepath(self):
        for key, val in self.filepaths.items():
            if val == "None":
                self.filepaths[key] = None
            else:
                self.filepaths[key] = self.get_full_filepath(val)
        if self.events:
            for event in self.events:
                for key, item in event.items():
                    if "filepath" in key:
                        event[key] = self.get_full_filepath(item)

    def export(self):
        with open(self.export_folder / 'parameters.yaml', 'w') as file:
            yaml.dump(self, file)

    def create_export_folder(self):
        if not os.path.isdir(paths.OUTPUT_FOLDER / self.scope):
            os.mkdir(paths.OUTPUT_FOLDER / self.scope)
        self.export_folder = paths.OUTPUT_FOLDER / self.scope / datetime.now().strftime('%Y%m%d_%H%M%S')
        os.mkdir(self.export_folder)

    def initialize_exports(self):
        if self.export_files and self.simulation_type != "criticality":
            self.create_export_folder()
            self.export()
            print(f'Output folder is {self.export_folder}')

    def adjust_logging_behavior(self):
        # Create logger
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs

        # Create a formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler (DEBUG and above)
        if self.export_files and self.simulation_type != "criticality":
            file_handler = logging.FileHandler(self.export_folder / 'exp.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Stream handler (INFO and above)
        console_handler = logging.StreamHandler()
        if self.logging_level == "debug":
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def add_variability_to_basic_cost(self):
        if not self.logistics['basic_cost_random']:
            self.logistics['nb_cost_profiles'] = 1
            self.logistics['basic_cost_variability'] = {mode: 0.0
                                                        for mode in self.logistics['basic_cost_variability'].keys()}
        self.logistics['basic_cost_profiles'] = {}
        drawn_costs = {mode: draw_lognormal_samples(mean_cost,
                                                    self.logistics['basic_cost_variability'][mode],
                                                    self.logistics['nb_cost_profiles'])
                       for mode, mean_cost in self.logistics['basic_cost'].items()}
        for i in range(self.logistics['nb_cost_profiles']):
            self.logistics['basic_cost_profiles'][i] = {
                mode: drawn_cost[i]
                for mode, drawn_cost in drawn_costs.items()
            }
