import json
import os

import pandas as pd


class Simulation(object):
    def __init__(self, simulation_type: str):
        self.type = simulation_type
        self.firm_data = []
        self.country_data = []
        self.household_data = []
        self.sc_network_data = []
        self.transport_network_data = []

    def export_agent_data(self, export_folder):
        with open(os.path.join(export_folder, 'firm_data.json'), 'w') as jsonfile:
            json.dump(self.firm_data, jsonfile)
        with open(os.path.join(export_folder, 'country_data.json'), 'w') as jsonfile:
            json.dump(self.country_data, jsonfile)
        with open(os.path.join(export_folder, 'household_data.json'), 'w') as jsonfile:
            json.dump(self.household_data, jsonfile)

    def export_transport_network_data(self, transport_edges, export_folder):
        flow_df = pd.DataFrame(self.transport_network_data)
        for time_step in flow_df['time_step'].unique():
            transport_edges_with_flows = pd.merge(
                transport_edges, flow_df[flow_df['time_step'] == time_step],
                how="left", on="id")
            transport_edges_with_flows.to_file(export_folder / f"transport_edges_with_flows_{time_step}.geojson",
                                               driver="GeoJSON", index=False)

    def calculate_and_export_summary_result(self):
        pass
