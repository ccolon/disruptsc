# Import modules
import cProfile
import csv
import logging
import pstats
import time
import argparse
from datetime import datetime

import paths
from disruptsc.model.basic_functions import mean_squared_distance
from disruptsc.model.caching_functions import generate_cache_parameters_from_command_line_argument, load_cached_model
from disruptsc.parameters import Parameters
from model.model import Model

profiler = cProfile.Profile()
profiler.enable()

parser = argparse.ArgumentParser(description="Mix positional and keyword arguments")
parser.add_argument("scope", type=str, help="Scope")
parser.add_argument("--cache", type=str, help="Caching behavior", required=False)
parser.add_argument("--duration", type=int, help="Disruption duration", required=False)
args = parser.parse_args()

# Start run
t0 = time.time()

# Retrieve scope
# scope = sys.argv[1]
scope = args.scope
logging.info(f'Simulation starting for {scope}')

# Generate cache parameters
cache_parameters = generate_cache_parameters_from_command_line_argument(args.cache)

# Import parameters
parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)

if args.duration:
    parameters.criticality['duration'] = args.duration

# Create the output folder and adjust logging behavior
parameters.initialize_exports()
parameters.adjust_logging_behavior()

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
# print("self.production_capacity", model.firms[0].production_capacity)
model.setup_logistic_routes(cached=cache_parameters['logistic_routes'])
# print("self.production_capacity", model.firms[0].production_capacity)

# Run model
if parameters.simulation_type == "initial_state":
    simulation = model.run_static()

elif parameters.simulation_type == "stationary_test":
    simulation = model.run_stationary_test()

elif parameters.simulation_type in ["event", "disruption"]:
    simulation = model.run_disruption()

elif parameters.simulation_type in ["event_mc", "disruption_mc"]:
    suffix = round(datetime.now().timestamp() * 1000)
    output_file = paths.OUTPUT_FOLDER / parameters.scope / f"disruption_{suffix}.csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        region_household_loss_labels = ['household_loss_' + region for region in model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in model.mrio.external_buying_countries]
        writer.writerow(["mc_repetition", "duration", "household_loss",
                         "country_loss"] + region_household_loss_labels + country_loss_labels)  # Writing the header

    logging.info(f"{parameters.mc_repetitions} Monte Carlo simulations")
    for i in range(parameters.mc_repetitions):
        logging.info(f"")
        logging.info(f"=============== Starting repetition #{i} ===============")
        model.setup_transport_network(cached=False)
        model.setup_agents(cached=False)
        model.setup_sc_network(cached=False)
        model.set_initial_conditions()
        model.setup_logistic_routes(cached=False)
        simulation = model.run_disruption(t_final=10)
        household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
        household_loss = sum(household_loss_per_region.values())
        country_loss_per_country = simulation.calculate_country_loss(per_countyr=True)
        country_loss = sum(country_loss_per_country.values())
        logging.info(f"Simulation terminated. "
                     f"Household loss: {int(household_loss)} {parameters.monetary_units_in_model}. "
                     f"Country loss: {int(country_loss)} {parameters.monetary_units_in_model}.")
        # Append the results to the CSV
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            household_loss_per_region_values = [household_loss_per_region.get(region, 0.0) for region in model.mrio.regions]
            country_loss_per_region_values = [country_loss_per_country.get(country, 0.0)
                                              for country in model.mrio.external_buying_countries]
            writer.writerow([i, household_loss, country_loss]
                            + household_loss_per_region_values + country_loss_per_region_values)


elif parameters.simulation_type in ['flow_calibration']:
    simulation = model.run_static()
    calibration_flows = simulation.report_annual_flow_specific_edges(parameters.flow_data, model.transport_edges,
                                                                     parameters.time_resolution, usd_or_ton='ton')
    print(calibration_flows)
    print(mean_squared_distance(calibration_flows, parameters.flow_data))

elif parameters.simulation_type == "criticality":
    suffix = round(datetime.now().timestamp() * 1000)
    output_file = paths.OUTPUT_FOLDER / parameters.scope / f"criticality_{suffix}.csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        region_household_loss_labels = ['household_loss_' + region for region in model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in model.mrio.external_buying_countries]
        writer.writerow(["edge_attr", parameters.criticality['attribute'], "duration", "household_loss", "country_loss"]
                        + region_household_loss_labels + country_loss_labels)  # Writing the header
    model.save_pickle(suffix)

    if parameters.criticality['attribute'] == "id":
        edges_to_test = parameters.criticality['edges']
        edges_to_test = {i: i for i in edges_to_test}
    else:
        # flat_list = list(chain.from_iterable(parameters.criticality['edges']
        #                                      if isinstance(parameters.criticality['edges'], list)
        #                                      else [parameters.criticality['edges']]))
        condition = model.transport_edges[parameters.criticality['attribute']].isin(parameters.criticality['edges'])
        edges_to_test = model.transport_edges.sort_values('id')[condition]
        edges_to_test = edges_to_test.set_index('id')[parameters.criticality['attribute']].to_dict()
    disruption_duration = parameters.criticality['duration']

    logging.info(f"Criticality simulation of {len(edges_to_test)} edges")
    for edge, attribute in edges_to_test.items():
        model = load_cached_model(suffix)
        simulation = model.run_criticality_disruption(edge, disruption_duration)
        household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
        household_loss = sum(household_loss_per_region.values())
        country_loss_per_country = simulation.calculate_country_loss(per_country=True)
        country_loss = sum(country_loss_per_country.values())
        logging.info(f"Simulation terminated. "
                     f"Household loss: {int(household_loss)} {parameters.monetary_units_in_model}. "
                     f"Country loss: {int(country_loss)} {parameters.monetary_units_in_model}.")
        # Append the results to the CSV
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            household_loss_per_region_values = [household_loss_per_region.get(region, 0.0) for region in model.mrio.regions]
            country_loss_per_region_values = [country_loss_per_country.get(country, 0.0)
                                              for country in model.mrio.external_buying_countries]
            writer.writerow([edge, attribute, parameters.criticality['duration'], household_loss, country_loss] +
                            household_loss_per_region_values + country_loss_per_region_values)

else:
    raise ValueError('Unimplemented simulation type chosen')

if parameters.export_files and parameters.simulation_type != "criticality":
    simulation.export_agent_data(parameters.export_folder)
    simulation.export_transport_network_data(model.transport_edges, parameters.export_folder)
    simulation.calculate_and_export_summary_result(model.sc_network, model.household_table,
                                                   parameters.monetary_units_in_model, parameters.export_folder)

logging.info(f"End of simulation, running time {time.time() - t0}")

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()