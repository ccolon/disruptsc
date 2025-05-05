# Import modules
import cProfile
import csv
import logging
import pstats
import time
import argparse
from datetime import datetime

import pandas as pd

import paths
from disruptsc.model.basic_functions import mean_squared_distance
from disruptsc.model.caching_functions import generate_cache_parameters_from_command_line_argument, load_cached_model
from disruptsc.parameters import Parameters
from model.model import Model

profiler = cProfile.Profile()
profiler.enable()

parser = argparse.ArgumentParser(description="Mix positional and keyword arguments")
parser.add_argument("scope", type=str, help="Scope")
parser.add_argument("--cache", type=str, help="Caching behavior")
parser.add_argument("--duration", type=int, help="Disruption duration")
parser.add_argument("--io_cutoff", type=float, help="IO cutoff")
# parser.add_argument("--with-transport", help="Deactivate transport completely", action='store_true')
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
if args.io_cutoff:
    parameters.io_cutoff = args.io_cutoff

if args.duration:
    parameters.criticality['duration'] = args.duration

# Create the output folder and adjust logging behavior
parameters.initialize_exports()
parameters.adjust_logging_behavior()

# Initialize model
model = Model(parameters)
model.setup_transport_network(cache_parameters['transport_network'], parameters.with_transport)
if parameters.export_files and parameters.simulation_type != "criticality" and parameters.with_transport:
    model.export_transport_nodes_edges()
model.setup_agents(cache_parameters['agents'])
if parameters.export_files and parameters.simulation_type != "criticality":
    model.export_agent_tables()
model.setup_sc_network(cache_parameters['sc_network'])
model.set_initial_conditions()
model.setup_logistic_routes(cache_parameters['logistic_routes'], parameters.with_transport)


# Run model
if parameters.simulation_type == "initial_state":
    simulation = model.run_static()

elif parameters.simulation_type == "initial_state_mc":
    flow_dfs = {}
    model.setup_transport_network(cached=False)
    model.setup_agents(cached=False)
    model.save_pickle('initial_state_mc')
    for i in range(parameters.mc_repetitions):
        logging.info(f"")
        logging.info(f"=============== Starting repetition #{i} ===============")
        model = load_cached_model("initial_state_mc")
        model.shuffle_logistic_costs()
        model.setup_sc_network(cached=False)
        model.set_initial_conditions()
        model.setup_logistic_routes(cached=False)
        simulation = model.run_static()
        flow_df = pd.DataFrame(simulation.transport_network_data)
        flow_df = flow_df[(flow_df['flow_total'] > 0) & (flow_df['time_step'] == 0)]
        flow_dfs[i] = flow_df
        flow_df.to_csv(parameters.export_folder / f"flow_df_{i}.csv")
    mean_flows = pd.concat(flow_dfs.values())
    mean_flows = mean_flows.groupby(mean_flows.index).mean()
    transport_edges_with_flows = pd.merge(model.transport_edges.drop(columns=["node_tuple"]),
                                          mean_flows, how="left", on="id")
    transport_edges_with_flows.to_file(parameters.export_folder / f"transport_edges_with_flows.geojson",
                                               driver="GeoJSON", index=False)


elif parameters.simulation_type == "stationary_test":
    simulation = model.run_stationary_test()

elif parameters.simulation_type in ["event", "disruption"]:
    simulation = model.run_disruption(t_final=parameters.t_final)

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


elif parameters.simulation_type in ["ad_hoc"]:
    suffix = round(datetime.now().timestamp() * 1000)
    output_file = paths.OUTPUT_FOLDER / parameters.scope / f"disruption_{suffix}.csv"
    periods = [30, 90, 180]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        household_loss_labels = ['household_loss_' + str(period) for period in periods]
        writer.writerow(["sectors", "duration", "household_loss"] + household_loss_labels + ["country_loss"])  # Writing the header

    model.save_pickle(suffix)

    disrupted_sector_list = ['all', ['ADM'], ['ADP'], ['ASO'], ['CAR'], ['CIN'], ['COM'], ['CON'], ['EDU'], ['ELE'], ['FIN'], ['FRT'], ['FRV'], ['INM'], ['MAQ'], ['MIP'], ['PES'], ['PPR'], ['QU2'], ['REF'], ['REP'], ['RES'], ['SAL'], ['SIL'], ['TEL'], ['TRA'], ('AGU', 'AYG'), ('AGU', 'BAL'), ('AGU', 'CEM'), ('AGU', 'CER'), ('AGU', 'DEM'), ('AGU', 'DOM'), ('AGU', 'GAN'), ('AGU', 'HIL'), ('AGU', 'LAC'), ('AGU', 'MAD'), ('AGU', 'MAN'), ('AGU', 'MOL'), ('AGU', 'MUE'), ('AGU', 'PAN'), ('AGU', 'PAP'), ('AGU', 'PLS'), ('AGU', 'SEG'), ('AGU', 'VES'), ('ALD', 'BAL'), ('ALD', 'CEM'), ('ALD', 'CER'), ('ALD', 'DOM'), ('ALD', 'GAN'), ('ALD', 'MAD'), ('ALD', 'MAN'), ('ALD', 'MOL'), ('ALD', 'PAP'), ('ALD', 'SEG'), ('AYG', 'BAL'), ('AYG', 'CEM'), ('AYG', 'CER'), ('AYG', 'DEM'), ('AYG', 'DOM'), ('AYG', 'GAN'), ('AYG', 'HIL'), ('AYG', 'LAC'), ('AYG', 'MAD'), ('AYG', 'MAN'), ('AYG', 'MOL'), ('AYG', 'MUE'), ('AYG', 'PAN'), ('AYG', 'PAP'), ('AYG', 'PLS'), ('AYG', 'SEG'), ('AYG', 'VES'), ('AZU', 'BAL'), ('AZU', 'CEM'), ('AZU', 'CER'), ('BAL', 'BNA'), ('BAL', 'CAN'), ('BAL', 'CAU'), ('BAL', 'CEM'), ('BAL', 'CER'), ('BAL', 'CHO'), ('BAL', 'CUE'), ('BAL', 'CUL'), ('BAL', 'DEM'), ('BAL', 'DOM'), ('BAL', 'FID'), ('BAL', 'GAN'), ('BAL', 'HIL'), ('BAL', 'HOT'), ('BAL', 'LAC'), ('BAL', 'MAD'), ('BAL', 'MAN'), ('BAL', 'MET'), ('BAL', 'MOL'), ('BAL', 'MUE'), ('BAL', 'PAN'), ('BAL', 'PAP'), ('BAL', 'PLS'), ('BAL', 'POS'), ('BAL', 'QU1'), ('BAL', 'SEG'), ('BAL', 'TAB'), ('BAL', 'VES'), ('BAL', 'VID'), ('BNA', 'CEM'), ('BNA', 'CER'), ('BNA', 'DEM'), ('BNA', 'DOM'), ('BNA', 'GAN'), ('BNA', 'MAD'), ('BNA', 'MAN'), ('BNA', 'MOL'), ('BNA', 'PAP'), ('BNA', 'SEG'), ('CAN', 'CEM'), ('CAN', 'CER'), ('CAU', 'CEM'), ('CAU', 'CER'), ('CEM', 'CER'), ('CEM', 'CHO'), ('CEM', 'CUE'), ('CEM', 'CUL'), ('CEM', 'DEM'), ('CEM', 'DOM'), ('CEM', 'FID'), ('CEM', 'GAN'), ('CEM', 'HIL'), ('CEM', 'HOT'), ('CEM', 'LAC'), ('CEM', 'MAD'), ('CEM', 'MAN'), ('CEM', 'MET'), ('CEM', 'MOL'), ('CEM', 'MUE'), ('CEM', 'PAN'), ('CEM', 'PAP'), ('CEM', 'PLS'), ('CEM', 'POS'), ('CEM', 'QU1'), ('CEM', 'SEG'), ('CEM', 'TAB'), ('CEM', 'VES'), ('CEM', 'VID'), ('CER', 'CHO'), ('CER', 'CUE'), ('CER', 'CUL'), ('CER', 'DEM'), ('CER', 'DOM'), ('CER', 'GAN'), ('CER', 'HIL'), ('CER', 'HOT'), ('CER', 'LAC'), ('CER', 'MAD'), ('CER', 'MAN'), ('CER', 'MET'), ('CER', 'MOL'), ('CER', 'MUE'), ('CER', 'PAN'), ('CER', 'PAP'), ('CER', 'PLS'), ('CER', 'POS'), ('CER', 'QU1'), ('CER', 'SEG'), ('CER', 'VES'), ('CER', 'VID'), ('CUL', 'GAN'), ('CUL', 'MAD'), ('CUL', 'MAN'), ('CUL', 'MOL'), ('CUL', 'PAP'), ('DEM', 'DOM'), ('DEM', 'GAN'), ('DEM', 'HIL'), ('DEM', 'HOT'), ('DEM', 'LAC'), ('DEM', 'MAD'), ('DEM', 'MAN'), ('DEM', 'MET'), ('DEM', 'MOL'), ('DEM', 'MUE'), ('DEM', 'PAN'), ('DEM', 'PAP'), ('DEM', 'PLS'), ('DEM', 'SEG'), ('DEM', 'VES'), ('DEM', 'VID'), ('DOM', 'GAN'), ('DOM', 'HIL'), ('DOM', 'HOT'), ('DOM', 'LAC'), ('DOM', 'MAD'), ('DOM', 'MAN'), ('DOM', 'MET'), ('DOM', 'MOL'), ('DOM', 'MUE'), ('DOM', 'PAN'), ('DOM', 'PAP'), ('DOM', 'PLS'), ('DOM', 'SEG'), ('DOM', 'VES'), ('DOM', 'VID'), ('GAN', 'HIL'), ('GAN', 'HOT'), ('GAN', 'LAC'), ('GAN', 'MAD'), ('GAN', 'MAN'), ('GAN', 'MET'), ('GAN', 'MOL'), ('GAN', 'MUE'), ('GAN', 'PAN'), ('GAN', 'PAP'), ('GAN', 'PLS'), ('GAN', 'QU1'), ('GAN', 'SEG'), ('GAN', 'VES'), ('GAN', 'VID'), ('HIL', 'MAD'), ('HIL', 'MAN'), ('HIL', 'MOL'), ('HIL', 'PAN'), ('HIL', 'PAP'), ('HIL', 'SEG'), ('HIL', 'VES'), ('HOT', 'MAD'), ('HOT', 'MAN'), ('HOT', 'MOL'), ('HOT', 'PAP'), ('HOT', 'SEG'), ('LAC', 'MAD'), ('LAC', 'MAN'), ('LAC', 'MOL'), ('LAC', 'PAN'), ('LAC', 'PAP'), ('LAC', 'SEG'), ('MAD', 'MAN'), ('MAD', 'MET'), ('MAD', 'MOL'), ('MAD', 'MUE'), ('MAD', 'PAN'), ('MAD', 'PAP'), ('MAD', 'PLS'), ('MAD', 'SEG'), ('MAD', 'VES'), ('MAD', 'VID'), ('MAN', 'MET'), ('MAN', 'MOL'), ('MAN', 'MUE'), ('MAN', 'PAN'), ('MAN', 'PAP'), ('MAN', 'PLS'), ('MAN', 'SEG'), ('MAN', 'VES'), ('MAN', 'VID'), ('MET', 'MOL'), ('MET', 'PAN'), ('MET', 'PAP'), ('MET', 'SEG'), ('MOL', 'MUE'), ('MOL', 'PAN'), ('MOL', 'PAP'), ('MOL', 'PLS'), ('MOL', 'SEG'), ('MOL', 'VES'), ('MOL', 'VID'), ('MUE', 'PAN'), ('MUE', 'PAP'), ('MUE', 'SEG'), ('MUE', 'VES'), ('PAN', 'PAP'), ('PAN', 'PLS'), ('PAN', 'SEG'), ('PAN', 'VES'), ('PAP', 'PLS'), ('PAP', 'SEG'), ('PAP', 'VES'), ('PAP', 'VID'), ('PLS', 'SEG'), ('PLS', 'VES'), ('SEG', 'VES'), ('SEG', 'VID')]

    present_sectors = list(set(model.firms.get_properties('region_sector', 'list')))
    logging.info(f"{parameters.mc_repetitions} Monte Carlo simulations")
    for disrupted_sectors in disrupted_sector_list:
        if disrupted_sectors == 'all':
            disrupted_sectors = present_sectors
        disrupted_sectors = ['ECU_'+sector for sector in disrupted_sectors]
        if any([sector not in present_sectors for sector in disrupted_sectors]):
            continue
        logging.info(f"")
        logging.info(f"=============== Disrupting sector #{disrupted_sectors} ===============")
        model = load_cached_model(suffix)
        model.parameters.events[0]['region_sectors'] = disrupted_sectors
        simulation = model.run_disruption(t_final=periods[-1])
        household_loss_per_periods = simulation.calculate_household_loss(model.household_table, periods=periods)
        household_loss = household_loss_per_periods[periods[-1]]
        country_loss = simulation.calculate_country_loss()

        logging.info(f"Simulation terminated. "
                     f"Household loss: {household_loss_per_periods}. "
                     f"Country loss: {int(country_loss)} {parameters.monetary_units_in_model}.")
        # Append the results to the CSV
        with open(output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["_".join(disrupted_sectors), household_loss, country_loss]
                            + list(household_loss_per_periods.values()) + [country_loss])

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
                        + region_household_loss_labels + country_loss_labels + ['geometry'])  # Writing the header
    model.save_pickle(suffix)

    if parameters.criticality['attribute'] == "top_flow":
        simulation = model.run_static()
        flow_df = pd.DataFrame(simulation.transport_network_data)
        flows = pd.merge(model.transport_edges, flow_df[flow_df['time_step'] == 0],
                                              how="left", on="id")
        flows = flows[flows['flow_total'] > 0]
        flows = flows[flows['type'].isin(['roads', 'railways'])]
        flows = flows.sort_values(by='flow_total', ascending=False)
        total = flows['flow_total'].sum()
        flows['cumulative_share'] = flows['flow_total'].cumsum() / total
        top_df = flows[flows['cumulative_share'] <= 0.9]
        edges_to_test = top_df.set_index('id')['flow_total'].to_dict()

    elif parameters.criticality['attribute'] == "id":
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

    logging.info(f"")
    logging.info(f"")
    logging.info(f"========== Criticality simulation of {len(edges_to_test)} edges ==========")
    for edge, attribute in edges_to_test.items():
        logging.info(f"")
        logging.info(f"=== Edge {edge} ====")
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
            geometry = model.transport_edges.loc[edge, 'geometry']
            writer.writerow([edge, attribute, parameters.criticality['duration'], household_loss, country_loss] +
                            household_loss_per_region_values + country_loss_per_region_values + [geometry])

else:
    raise ValueError('Unimplemented simulation type chosen')

if parameters.export_files and parameters.simulation_type not in ["criticality", "initial_state_mc"]:
    simulation.export_agent_data(parameters.export_folder)
    simulation.export_transport_network_data(model.transport_edges, parameters.export_folder)
    simulation.calculate_and_export_summary_result(model.sc_network, model.household_table,
                                                   parameters.monetary_units_in_model, parameters.export_folder)

logging.info(f"End of simulation, running time {time.time() - t0}")

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
# stats.print_stats()