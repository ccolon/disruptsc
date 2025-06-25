import json
import logging
import os
from pathlib import Path
import pandas as pd
import geopandas as gpd

from disruptsc.network.sc_network import ScNetwork


class Simulation(object):
    def __init__(self, simulation_type: str):
        admissible_types = ["initial_state", "event", "disruption", "stationary_test", "criticality"]
        if simulation_type not in admissible_types:
            raise ValueError(f"Simulation type should be {admissible_types}")
        self.type = simulation_type
        self.firm_data = []
        self.country_data = []
        self.household_data = []
        self.sc_network_data = []
        self.transport_network_data = []

    def export_agent_data(self, export_folder):
        logging.info(f'Exporting agent data to {export_folder}')
        with open(os.path.join(export_folder, 'firm_data.json'), 'w') as jsonfile:
            json.dump(self.firm_data, jsonfile)
        with open(os.path.join(export_folder, 'country_data.json'), 'w') as jsonfile:
            json.dump(self.country_data, jsonfile)
        with open(os.path.join(export_folder, 'household_data.json'), 'w') as jsonfile:
            json.dump(self.household_data, jsonfile)

    def export_transport_network_data(self, transport_edges: gpd.GeoDataFrame, export_folder: Path):
        if export_folder:
            logging.info(f'Exporting transport network data to {export_folder}')
            flow_df = pd.DataFrame(self.transport_network_data)
            flow_df = flow_df[flow_df['flow_total'] > 0]
            for time_step in flow_df['time_step'].unique():
                transport_edges_with_flows = pd.merge(
                    transport_edges.drop(columns=["node_tuple"]), flow_df[flow_df['time_step'] == time_step],
                    how="left", on="id")
                transport_edges_with_flows.to_file(export_folder / f"transport_edges_with_flows_{time_step}.geojson",
                                                   driver="GeoJSON", index=False)

    def calculate_and_export_summary_result(self, sc_network: ScNetwork, household_table: pd.DataFrame,
                                            monetary_unit_in_model: str, export_folder: Path):
        if self.type == "initial_state":
            # export io matrix
            logging.info(f'Exporting resulting IO matrix to {export_folder}')
            sc_network.calculate_io_matrix().to_csv(export_folder / "io_table.csv")
            logging.info(f'Exporting edgelist to {export_folder}')
            sc_network.generate_edge_list().to_csv(export_folder / "sc_network_edgelist.csv")

        else:# self.type == "event":
            # export loss time series for households
            household_result_table = pd.DataFrame(self.household_data)
            loss_per_region_sector_time = household_result_table.groupby('household').apply(
                self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
            household_table['id'] = 'hh_' + household_table['id'].astype(str)
            loss_per_region_sector_time['region'] = loss_per_region_sector_time['household'].map(
                household_table.set_index('id')['region'])
            loss_per_region_sector_time = \
                loss_per_region_sector_time.groupby(['region', 'sector', 'time_step'], as_index=False)['loss'].sum()
            if export_folder:
                logging.info(f'Exporting loss time series of households per region sector to {export_folder}')
                loss_per_region_sector_time.to_csv(export_folder / "loss_per_region_sector_time.csv", index=False)
            household_loss = loss_per_region_sector_time['loss'].sum()
            # export loss time series for countries
            country_result_table = pd.DataFrame(self.country_data)
            country_result_table['loss'] = country_result_table['extra_spending'] \
                                           + country_result_table['consumption_loss']
            country_result_table = country_result_table[['time_step', 'country', 'loss']]
            country_loss = country_result_table['loss'].sum()
            if export_folder:
                logging.info(f'Exporting loss time series of countries to {export_folder}')
                country_result_table.to_csv(export_folder / "loss_per_country.csv", index=False)
            logging.info(f"Cumulated household loss: {household_loss:,.2f} {monetary_unit_in_model}")
            logging.info(f"Cumulated country loss: {country_loss:,.2f} {monetary_unit_in_model}")
            # Export summary
            total_loss = pd.DataFrame({"households": household_loss, "countries": country_loss}, index=[0])
            if export_folder:
                total_loss.to_csv(export_folder / "loss_summary.csv", index=False)

        # elif self.type == "criticality":
        #     household_loss = self.calculate_household_loss()
        #     country_loss = self.calculate_country_loss()

    def get_flow_specific_edges(self, edge_names: list, transport_edges: gpd.GeoDataFrame, usd_or_ton: str = 'usd'):
        flow_df = pd.DataFrame(self.transport_network_data)
        specific_edges_id_to_name = transport_edges.loc[transport_edges['name'].isin(edge_names), ['name', 'id']]
        specific_edges_id_to_name = specific_edges_id_to_name.set_index('id')['name'].to_dict()
        flow_df = flow_df[flow_df['id'].isin(list(specific_edges_id_to_name.keys()))].copy()
        flow_df['name'] = flow_df['id'].map(specific_edges_id_to_name)
        col_to_report = 'flow_total' if usd_or_ton == 'usd' else 'flow_total_tons'
        return flow_df.set_index('name')[col_to_report].to_dict()

    def report_annual_flow_specific_edges(self, edge_names: list, transport_edges: gpd.GeoDataFrame,
                                          time_resolution: str, usd_or_ton: str = 'usd'):
        flows = self.get_flow_specific_edges(edge_names, transport_edges, usd_or_ton)
        periods = {'day': 365, 'week': 52, 'month': 12, 'year': 1}
        flows = pd.Series(flows) * periods[time_resolution]
        return flows.to_dict()

    @staticmethod
    def summarize_results_one_household(household_result_table_one_household):
        extra_spending_per_sector_table = pd.DataFrame(
            household_result_table_one_household.set_index('time_step')['extra_spending_per_sector'].to_dict()
        ).transpose()
        consumption_loss_per_sector_table = pd.DataFrame(
            household_result_table_one_household.set_index('time_step')['consumption_loss_per_sector'].to_dict()
        ).transpose()
        loss_per_sector = extra_spending_per_sector_table + consumption_loss_per_sector_table
        result = loss_per_sector.stack().reset_index()
        result.columns = ['time_step', 'sector', 'loss']
        return result

    def calculate_household_loss(self, household_table: pd.DataFrame, per_region=False, periods=None):
        household_result_table = pd.DataFrame(self.household_data)
        loss_per_region_sector_time = household_result_table.groupby('household').apply(
            self.summarize_results_one_household).reset_index().drop(columns=['level_1'])
        loss_per_region_sector_time['origin_region'] = loss_per_region_sector_time['sector'].str.extract(r'([A-Z]*)_')
        loss_per_region_sector_time['household_region'] = loss_per_region_sector_time['household'].map(
            household_table.set_index('household')['region'])
        if per_region:
            return loss_per_region_sector_time.groupby('household_region')['loss'].sum().to_dict()
        elif isinstance(periods, list):
            household_result_table['total_loss'] = household_result_table['extra_spending'] + household_result_table['consumption_loss']
            ts = household_result_table.groupby('time_step')['total_loss'].sum()
            baseline = household_result_table.loc[household_result_table['time_step'] == 0, 'tot_consumption'].sum()
            return {
                period: ts[:period].sum() / (baseline * period)
                for period in periods
            }
        else:
            return loss_per_region_sector_time['loss'].sum()

    def calculate_country_loss(self, per_country=False):
        country_result_table = pd.DataFrame(self.country_data)
        country_result_table['loss'] = country_result_table['extra_spending'] \
                                       + country_result_table['consumption_loss']
        country_result_table = country_result_table[['time_step', 'country', 'loss']]
        if per_country:
            return country_result_table.groupby('country')['loss'].sum().to_dict()
        else:
            return country_result_table['loss'].sum()
