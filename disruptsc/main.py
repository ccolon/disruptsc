# Import modules
import cProfile
import csv
import logging
import pstats
import sys
import time
from datetime import datetime

import paths
from disruptsc.model.caching_functions import generate_cache_parameters_from_command_line_argument, load_cached_model
from disruptsc.parameters import Parameters
from disruptsc.simulation.handling_functions import check_script_call
from model.model import Model

# profiler = cProfile.Profile()
# profiler.enable()

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

parameters.adjust_logging_behavior(parameters.simulation_type == "criticality")

# Initialize model
model = Model(parameters)
model.setup_transport_network(cached=cache_parameters['transport_network'])
if parameters.export_files and parameters.simulation_type != "criticality":
    model.export_transport_nodes_edges()
model.setup_agents(cached=cache_parameters['agents'])
if parameters.export_files and parameters.simulation_type != "criticality":
    model.export_agent_tables()
model.setup_sc_network(cached=cache_parameters['sc_network'])
model.set_initial_conditions()
model.setup_logistic_routes(cached=cache_parameters['logistic_routes'])

# Run model
if parameters.simulation_type == "initial_state":
    simulation = model.run_static()

elif parameters.simulation_type == "disruption":
    simulation = model.run_disruption()

elif parameters.simulation_type == "criticality":
    suffix = round(datetime.now().timestamp() * 1000)
    output_file = paths.OUTPUT_FOLDER / parameters.scope / f"criticality_{suffix}.csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["edge", "household_loss", "country_loss"])  # Writing the header
    model.save_pickle(suffix)
    edges_to_test = parameters.criticality['edges']
    disruption_duration = parameters.criticality['duration']

    for edge in edges_to_test:
        model = load_cached_model(suffix)
        simulation = model.run_criticality_disruption(edge, disruption_duration)
        household_loss = simulation.calculate_household_loss()
        country_loss = simulation.calculate_country_loss()
        # Append the results to the CSV
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([edge, household_loss, country_loss])

else:
    raise ValueError('Unimplemented simulation type chosen')

if parameters.export_files:
    simulation.export_agent_data(parameters.export_folder)
    simulation.export_transport_network_data(model.transport_edges, parameters.export_folder)
    simulation.calculate_and_export_summary_result(model.sc_network, model.household_table,
                                                   parameters.monetary_units_in_model, parameters.export_folder)

logging.info(f"End of simulation, running time {time.time() - t0}")

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()
