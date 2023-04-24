import json
import os


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
