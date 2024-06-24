# Import modules
import logging
import sys
import time
import paths
from code.model.caching_functions import generate_cache_parameters_from_command_line_argument
from code.parameters import Parameters
from code.simulation.handling_functions import check_script_call
from model.model import Model

# Start run
t0 = time.time()

# Check that the script is called correctly
check_script_call(sys.argv)

# Retrieve scope
scope = sys.argv[1]
logging.info(f'Simulation starting for {scope}')

# Generate cache parameters
cache_parameters = generate_cache_parameters_from_command_line_argument(sys.argv)

# Import parameters
parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)

# Create the output folder and adjust logging behavior
if parameters.export_files:
    parameters.create_export_folder()
    parameters.export()
    logging.info(f'Output folder is {parameters.export_folder}')

parameters.adjust_logging_behavior()

# Initialize model
model = Model(parameters)
model.setup_transport_network(cached=cache_parameters['transport_network'])
if parameters.export_files:
    model.export_transport_nodes_edges()
model.setup_agents(cached=cache_parameters['agents'])
if parameters.export_files:
    model.export_agent_tables()
model.setup_sc_network(cached=cache_parameters['sc_network'])
model.set_initial_conditions()
model.setup_logistic_routes(cached=cache_parameters['logistic_routes'])

# Run model
if parameters.simulation_type == "initial_state":
    simulation = model.run_static()

elif parameters.simulation_type == "disruption":
    simulation = model.run_disruption()

else:
    raise ValueError('Unimplemented simulation type chosen')

if parameters.export_files:
    simulation.export_agent_data(parameters.export_folder)
    simulation.export_transport_network_data(model.transport_edges, parameters.export_folder)
    simulation.calculate_and_export_summary_result(model.sc_network, model.household_table,
                                                   parameters.monetary_units_in_model, parameters.export_folder)

logging.info("End of simulation")
