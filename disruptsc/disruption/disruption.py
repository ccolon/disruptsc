from itertools import chain
from typing import TYPE_CHECKING

import logging
from collections import UserList
from pathlib import Path

import geopandas
import pandas
import pandas as pd

from disruptsc.parameters import EPSILON, import_code

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.network.transport_network import TransportNetwork
    from disruptsc.model.model import Model


class ReconstructionMarket:
    def __init__(self, reconstruction_target_time: int, capital_input_mix: dict):
        self.reconstruction_target_time = reconstruction_target_time
        self.capital_input_mix = capital_input_mix
        self.aggregate_demand = 0
        self.aggregate_demand_per_sector = {}
        self.demand_to_firm_per_sector = {}

    def send_orders(self, firms: "Firms"):
        for sector, demand_to_firm_this_sector in self.demand_to_firm_per_sector.items():
            for pid, reconstruction_demand in demand_to_firm_this_sector.items():
                firms[pid].reconstruction_demand = reconstruction_demand
                firms[pid].add_reconstruction_order_to_order_book()

    def distribute_new_capital(self, firms: "Firms"):
        # Retrieve production
        print("total capital demanded:", self.aggregate_demand)
        amount_produced_per_sector = {}
        for sector in self.capital_input_mix.keys():
            if sector == import_code:
                amount_produced_per_sector[sector] = self.aggregate_demand_per_sector[sector]
            else:
                amount_produced_per_sector[sector] = sum([firm.reconstruction_produced
                                                          for firm in firms.filter_by_sector(sector).values()])
        print("amount_produced_per_sector:", amount_produced_per_sector)
        # Produce (we suppose that what is not used disappear, no stock of unfinished capital)
        new_capital_produced = min([amount_produced_per_sector[sector] / weight
                                    for sector, weight in self.capital_input_mix.items()])
        print("new_capital_produced:", new_capital_produced)
        # Send new capital to firm
        for firm in firms.values():
            firm.capital_destroyed -= (firm.capital_demanded / self.aggregate_demand) * new_capital_produced
        print("total capital destroyed:", firms.sum("capital_destroyed"))
        print("average production capacity:", firms.mean("current_production_capacity"))

    def evaluate_demand_to_firm(self, firms: "Firms"):
        # Retrieve the demand of each firm, and translate it into a demand for certain inputs (sectors)
        for firm in firms.values():
            firm.capital_demanded = firm.capital_destroyed / self.reconstruction_target_time
        self.aggregate_demand = sum([firm.capital_demanded for firm in firms.values()])
        self.aggregate_demand_per_sector = {sector: weight * self.aggregate_demand
                                            for sector, weight in self.capital_input_mix.items()}

        # Get potential supply per sector and evaluate whether demand needs to be rationed
        rationing_per_sector = {}
        total_supply_per_sector = {}
        potential_supply_per_firm_per_sector = {}
        for sector in self.capital_input_mix.keys():
            if sector == import_code:  # No constraints for imported products
                total_supply_per_sector[sector] = self.aggregate_demand_per_sector[sector]
                rationing_per_sector[sector] = 1
            else:
                potential_supply_per_firm_per_sector[sector] = {pid: firm.get_spare_production_potential()
                                                                for pid, firm in firms.items()
                                                                if firm.sector == sector}
                total_supply_per_sector[sector] = sum(potential_supply_per_firm_per_sector[sector].values())
                rationing_per_sector[sector] = min(1, total_supply_per_sector[sector]
                                                   / self.aggregate_demand_per_sector[sector])
        rationing = min(list(rationing_per_sector.values()))
        if rationing > 1 - EPSILON:
            logging.info("Reconstruction market: There is no rationing")
        else:
            logging.info(f"Reconstruction market: Due to limited capacity, "
                         f"supply for reconstruction is {rationing:.2%} of demand")

        # Evaluate actual demand per firm
        for sector in self.capital_input_mix.keys():
            if sector == import_code:
                self.demand_to_firm_per_sector[sector] = {}
            else:
                if rationing < EPSILON:
                    self.demand_to_firm_per_sector[sector] = {}
                else:
                    adjusted_demand_this_sector = self.aggregate_demand_per_sector[sector] * rationing
                    firm_weight = {pid: potential_supply / total_supply_per_sector[sector]
                                   for pid, potential_supply in potential_supply_per_firm_per_sector[sector].items()}
                    self.demand_to_firm_per_sector[sector] = {pid: weight * adjusted_demand_this_sector
                                                              for pid, weight in firm_weight.items()}


class Recovery:
    def __init__(self, duration: int, shape: str):
        self.duration = duration
        self.shape = shape


class TransportDisruption(dict):
    def __init__(self, description: dict, recovery: Recovery | None = None, start_time: int = 1):
        self.start_time = start_time
        self.recovery = recovery
        if description is not None:
            for key, value in description.items():
                self.__setitem__(key, value)
        super().__init__()

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise KeyError("Key must be a int: the id of the transport edge_attr to be disrupted")
        if not isinstance(value, float):
            raise ValueError("Value must be a float: the fraction of lost capacity")
        super().__setitem__(key, value)

    def __repr__(self):
        return f"EventDict(start_time={self.start_time}, data={super().__repr__()})"

    def log_info(self):
        if self.recovery:
            logging.info(f"{len(self)} transport edges are disrupted at {self.start_time} time steps. "
                         f"They recover after {self.recovery.duration} time steps.")
        else:
            logging.info(f"{len(self)} transport edges are disrupted at {self.start_time} time steps. "
                         f"There is no recovery.")

    @classmethod
    def from_edge_attributes(cls, edges: geopandas.GeoDataFrame, attribute: str, values: list):
        # we do a special case for the disruption attribute
        # for which we check if the attribute contains one of the value
        if attribute == "disruption":
            condition = [edges[attribute].str.contains(value) for value in values]
            condition = pandas.concat(condition, axis=1)
            condition = condition.any(axis=1)
        else:
            condition = edges[attribute].isin(values)
        item_ids = edges.sort_values('id').loc[condition, 'id'].tolist()
        description = pd.Series(1.0, index=item_ids).to_dict()

        return cls(
            description=description,
        )

    def implement(self, transport_network: "TransportNetwork"):
        for edge in transport_network.edges:
            edge_id = transport_network[edge[0]][edge[1]]['id']
            if edge_id in self.keys():
                transport_network.disrupt_one_edge(edge, self[edge_id], self.recovery.duration)


class CapitalDestruction(dict):
    def __init__(self, description: dict, recovery: Recovery | None = None, start_time: int = 1,
                 reconstruction_market: bool = False):
        self.start_time = start_time
        self.recovery = recovery
        self.reconstruction_market = reconstruction_market
        if description is not None:
            for key, value in description.items():
                self.__setitem__(key, value)
        super().__init__()

    def __setitem__(self, key, value):
        if not isinstance(key, int):
            raise KeyError("Key must be an int: the id of the firm")
        if not isinstance(value, float):
            raise ValueError("Value must be a float: the amount of destroyed capital")
        super().__setitem__(key, value)

    def __repr__(self):
        return f"EventDict(start_time={self.start_time}, data={super().__repr__()})"

    def log_info(self):
        logging.info(f"{len(self)} firms incur capital destruction at {self.start_time} time steps. "
                     f"There is {'a' if self.reconstruction_market else 'no'} reconstruction market.")
        if self.recovery:
            logging.info(f"There is exogenous recovery after {self.recovery.duration} time steps.")
        else:
            logging.info("There is no exogenous recovery.")

    @classmethod
    def from_region_sector_file(cls, filepath: Path, firm_table: pd.DataFrame, firm_list: "Firms",
                                input_units: str, target_units: str):
        df = pd.read_csv(filepath, dtype={'region': str, 'sector': str, 'destroyed_capital': float})
        units = {"USD": 1, "kUSD": 1e3, "mUSD": 1e6}
        df['destroyed_capital'] = df['destroyed_capital'] * units[input_units] / units[target_units]
        result = {}
        dic_destroyed_capital_per_region_sector = df.set_index(['region', 'sector'])['destroyed_capital'].to_dict()
        for region_sector, destroyed_capital in dic_destroyed_capital_per_region_sector.items():
            region, sector = region_sector
            firms = firm_table.loc[(firm_table['region'] == region) & (firm_table['sector'] == sector), "id"].to_list()
            if len(firms) == 0:
                logging.warning(f"In {region_sector}, destroyed capital is {destroyed_capital} {input_units} "
                                f"but there are no firm modeled")
            else:
                total_capital = sum([firm_list[firm_id].capital_initial for firm_id in firms])
                for firm_id in firms:
                    weight = firm_list[firm_id].capital_initial / total_capital
                    result[firm_id] = weight * destroyed_capital
        df = pd.merge(df, firm_table[['region', 'sector', 'id']], how='left', on=['region', 'sector'])
        total_destroyed_capital_in_data = df['destroyed_capital'].sum()
        total_destroyed_capital_in_model = sum(result.values())
        logging.info(f"Destroyed capital in data: {total_destroyed_capital_in_data}, "
                     f"Destroyed capital in model: {total_destroyed_capital_in_model}")

        return cls(
            description=result,
            recovery=None
        )

    @classmethod
    def from_firms_attributes(cls, destroyed_amount: float, filters: dict, firms: "Firms",
                              input_units: str, model_units: str):
        units = {"USD": 1, "kUSD": 1e3, "mUSD": 1e6}
        destroyed_amount = destroyed_amount * units[input_units] / units[model_units]

        affected_firms = firms.select_by_properties(filters)
        total_capital = sum([firm.capital_initial for firm in affected_firms.values()])
        if destroyed_amount > total_capital:
            logging.warning(f"Destroyed capital {destroyed_amount:.0f} larger than initial capital {total_capital:.0f}")
            description = {firm_id: firm.capital_initial for firm_id, firm in affected_firms.items()}
        else:
            description = {firm_id: firm.capital_initial / total_capital * destroyed_amount
                           for firm_id, firm in affected_firms.items()}
        return cls(
            description=description,
            recovery=None
        )

    def implement(self, firms: "Firms", model: "Model"):
        for firm_id, destroyed_capital in self.items():
            firms[firm_id].incur_capital_destruction(destroyed_capital)
        if self.reconstruction_market:
            model.reconstruction_market = ReconstructionMarket(reconstruction_target_time=30,
                                                               capital_input_mix={"CON": 0.7, "MAN": 0.2, "IMP": 0.1})


class DisruptionList(UserList):
    def __init__(self, disruption_list: list):
        super().__init__(disruption for disruption in disruption_list
                         if isinstance(disruption, CapitalDestruction) or isinstance(disruption, TransportDisruption))
        if len(disruption_list) > 0:
            self.start_time = min([disruption.start_time for disruption in disruption_list])
            self.end_time = max([disruption.start_time + 1 for disruption in disruption_list])
            # self.end_time = max([disruption.start_time + disruption.duration for disruption in disruption_list])
            # self.transport_nodes = [
            #     disruption.item_id
            #     for disruption in disruption_list
            #     if disruption.item_type == "transport_node"
            # ]
            # self.transport_edges = [
            #     disruption.item_id
            #     for disruption in disruption_list
            #     if disruption.item_type == "transport_edge"
            # ]
            # self.firms = [
            #     disruption.item_id
            #     for disruption in disruption_list
            #     if disruption.item_type == "firm"
            # ]
        else:
            self.start_time = 0
            self.end_time = 0

    @classmethod
    def from_events_parameter(
            cls,
            events: list,
            model_unit: str,
            edges: geopandas.GeoDataFrame,
            firm_table: pandas.DataFrame,
            firm_list: "Firms"
    ):
        event_list = []
        for event in events:
            # TODO: create a load_one_event fct, and factor
            if event['type'] == "capital_destruction":
                if event['description_type'] == "region_sector_file":
                    disruption_object = CapitalDestruction.from_region_sector_file(event['region_sector_filepath'],
                                                                                   firm_table,
                                                                                   firm_list,
                                                                                   input_units=event['unit'],
                                                                                   target_units=model_unit)
                    disruption_object.start_time = event["start_time"]
                    if "reconstruction_market" in event.keys():
                        disruption_object.reconstruction_market = event["reconstruction_market"]
                    event_list += [disruption_object]

                if event['description_type'] == "filter":
                    filters = {"region_sector": event['region_sectors']}
                    disruption_object = CapitalDestruction.from_firms_attributes(event['destroyed_capital'],
                                                                                 event['filter'], firm_list,
                                                                                 event['unit'], model_unit)
                    disruption_object.start_time = event["start_time"]
                    if "reconstruction_market" in event.keys():
                        disruption_object.reconstruction_market = event["reconstruction_market"]
                    event_list += [disruption_object]

            if event['type'] == "transport_disruption":
                if event['description_type'] == "edge_attributes":
                    # flat_list = list(chain.from_iterable(event['values']
                    #                         if isinstance(event['values'], list)
                    #                         else [event['values']]))
                    disruption_object = TransportDisruption.from_edge_attributes(
                        edges=edges,
                        attribute=event['attribute'],
                        values=event['values']
                    )
                    disruption_object.start_time = event["start_time"]
                    disruption_object.recovery = Recovery(duration=event['duration'], shape="threshold")
                    event_list += [disruption_object]
        return cls(event_list)

    def log_info(self):
        logging.info(f'There are {len(self)} disruptions')
        for disruption in self:
            disruption.log_info()

    # def filter_type(self, selected_item_type):
    #     return DisruptionList([disruption for disruption in self if disruption.item_type == selected_item_type])

    def filter_start_time(self, selected_start_time):
        return DisruptionList([disruption for disruption in self if disruption.start_time == selected_start_time])

    def get_item_id_duration_reduction_dict(self) -> dict:
        return {
            disruption.item_id: {
                "duration": disruption.duration,
                "reduction": disruption.reduction
            }
            for disruption in self
        }

    def get_id_list(self) -> list:
        return [disruption.item_id for disruption in self]
