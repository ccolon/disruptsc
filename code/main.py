# Import modules
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from datetime import datetime
import importlib
import paths
from code.export_functions import exportParameters
from code.model.caching_functions import generate_cache_parameters_from_command_line
from code.parameters import Parameters
from model.model import Model

# Import functions and classes
# from export_functions import *

# Import parameters. It should be in this specific order.
parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER)

# Check that the script is called correctly
accepted_script_arguments: list[str] = [
    'same_transport_network_new_agents',
    'same_agents_new_sc_network',
    'same_sc_network_new_logistic_routes',
    'same_logistic_routes',
]
if len(sys.argv) > 2:
    raise ValueError('The script does not take more than 1 arguments')
if len(sys.argv) > 1:
    if sys.argv[1] not in accepted_script_arguments:
        raise ValueError("First argument " + sys.argv[1] + " is not valid.\
            Possible values are: " + ','.join(accepted_script_arguments))

# Generate cache parameters
cache_parameters: dict[str, bool] = {
    "transport_network": False,
    "agents": False,
    "sc_network": False,
    "logistic_routes": False
}
if len(sys.argv) > 1:
    cache_parameters = generate_cache_parameters_from_command_line(sys.argv[1])

# Start run
t0 = time.time()


def create_export_folder(main_output_folder: Path, input_folder_name: str) -> Path:
    if not os.path.isdir(main_output_folder / input_folder_name):
        os.mkdir(main_output_folder / input_folder_name)
    exp_folder = main_output_folder / input_folder_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    os.mkdir(exp_folder)
    exportParameters(exp_folder)
    return exp_folder


def adjust_logging_behavior(export: dict, exp_folder: Path, selected_logging_level: str):
    if selected_logging_level == "info":
        logging_level = logging.INFO
    else:
        logging_level = logging.DEBUG

    if export['log']:
        importlib.reload(logging)
        logging.basicConfig(
            filename=exp_folder / 'exp.log',
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


# We create the output folder
export_folder = create_export_folder(paths.OUTPUT_FOLDER, parameters.input_folder)

# Set logging parameters
adjust_logging_behavior(parameters.export, export_folder, parameters.logging_level)

logging.info(f'Simulation starting using {parameters.input_folder}, output folder is {export_folder}')

# Initialize model
model = Model(parameters)
model.setup_transport_network(cached=cache_parameters['transport_network'])
model.setup_agents(cached=cache_parameters['agents'])
model.setup_sc_network(cached=cache_parameters['sc_network'])
model.set_initial_conditions()
model.setup_logistic_routes(cached=cache_parameters['logistic_routes'])
exit()

logging.info("End of simulation")
