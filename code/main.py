# Import modules
import sys
import time
import yaml
from datetime import datetime
import importlib
import paths
import pickle
from model.model import Model

# Import functions and classes
from builder import *
from simulations import *
from export_functions import *
from class_observer import Observer

# Import parameters. It should be in this specific order.
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, project_path)
from parameter.parameters import *
from parameter.filepaths import *

# Check that the script is called correctly
accepted_script_arguments = [
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

# Start run
t0 = time.time()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


def create_export_folder(
        project_folder: str,
        export: dict,
        input_folder: str,
        timestamp: str
) -> str:
    if any(list(export.values())):
        exp_folder = os.path.join(project_folder, 'output', input_folder, timestamp)
        if not os.path.isdir(os.path.join(project_folder, 'output', input_folder)):
            os.mkdir(os.path.join(project_folder, 'output', input_folder))
        os.mkdir(exp_folder)
        exportParameters(exp_folder)

    else:
        exp_folder = None

    return exp_folder


def adjust_logging_behavior(project_folder: str, export: dict, exp_folder: str, logging_level):
    if export['log']:
        importlib.reload(logging)
        print(os.path.join(project_folder, exp_folder, 'exp.log'))
        logging.basicConfig(
            filename=os.path.join(project_folder, exp_folder, 'exp.log'),
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


# If there is sth to export, then we create the output folder
exp_folder = create_export_folder(paths.project_directory, export, input_folder, timestamp)

# Set logging parameters
adjust_logging_behavior(paths.project_directory, export, exp_folder, logging_level)

logging.info('Simulation ' + timestamp + ' starting using ' + input_folder + ' input data.')

# Create transport network
with open(filepaths['transport_parameters'], "r") as yaml_file:
    transport_params = yaml.load(yaml_file, Loader=yaml.FullLoader)

model = Model(parameters, filepaths)

model.setup_transport_network()

logging.info("End of simulation")
