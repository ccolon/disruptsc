from collections import UserList

import geopandas
import pandas


class Disruption:
    def __init__(self, start_time: int, duration: int, item_id: int | str, item_type: str, reduction: float):
        self.start_time = start_time
        self.duration = duration
        self.item_id = item_id
        self.item_type = item_type
        self.reduction = reduction  # capacity reduction

    @classmethod
    def from_transport_edge_attributes(
            cls,
            edges: geopandas.GeoDataFrame,
            attribute: str,
            values: list,
            start_time: int,
            duration: int
    ) -> list:
        # we do a special case for the disruption attribute
        # for which we check if the attribute contains one of the value
        if attribute == "disruption":
            condition = [edges[attribute].str.contains(value) for value in values]
            condition = pandas.concat(condition, axis=1)
            condition = condition.any(axis=1)
        else:
            condition = edges[attribute].isin(values)
        item_ids = edges.sort_values('id').loc[condition, 'id'].tolist()

        return [
            cls(
                start_time=start_time,
                duration=duration,
                item_id=item_id,
                item_type="transport_edge",
                reduction=1
            ) for item_id in item_ids
        ]

    @classmethod
    def from_sector_admin_unit(
            cls,
            firm_table: pandas.DataFrame,
            sectors: list,
            districts: list,
            start_time: int,
            duration: int,
            reduction: float
    ) -> list:
        cond_district = firm_table['admin_unit'].isin(districts)
        cond_sector = firm_table['sector'].isin(sectors)
        item_ids = firm_table.loc[
            cond_district & cond_sector,
            'id'
        ].tolist()

        return [
            cls(
                start_time=start_time,
                duration=duration,
                item_id=item_id,
                item_type="firm",
                reduction=reduction
            ) for item_id in item_ids
        ]

    @classmethod
    def from_disruption_description(
            cls,
            disruption_description: dict,
            edges: geopandas.GeoDataFrame,
            firm_table: pandas.DataFrame
    ) -> list:
        disruption_list_transport_edges = [
            Disruption.from_sector_admin_unit(
                firm_table=firm_table,
                sectors=disruption_event['sectors'],
                districts=disruption_event['admin_units'],
                start_time=disruption_event['start_time'],
                duration=disruption_event['duration'],
                reduction=disruption_event['production_capacity_reduction']
            ) for disruption_event in disruption_description['events']
            if disruption_event['item_type'] == "firms"
        ]
        disruption_list_transport_edges = [item for sublist in disruption_list_transport_edges for item in sublist]
        disruption_list_firms = [
            Disruption.from_transport_edge_attributes(
                edges=edges,
                attribute=disruption_event['attribute'],
                values=disruption_event['values'],
                start_time=disruption_event['start_time'],
                duration=disruption_event['duration']
            ) for disruption_event in disruption_description['events']
            if disruption_event['item_type'] == "transport_edges"
        ]
        disruption_list_firms = [item for sublist in disruption_list_firms for item in sublist]
        disruption_list = disruption_list_transport_edges + disruption_list_firms
        return disruption_list

    def print_info(self):
        print(f"Capacity of {self.item_type} {self.item_id} is reduced by {self.reduction*100}% \
              at time {self.start_time} during {self.duration} time steps")


class DisruptionList(UserList):
    def __init__(self, disruption_list: list):
        super().__init__(disruption for disruption in disruption_list if isinstance(disruption, Disruption))
        if len(disruption_list) > 0:
            self.start_time = min([disruption.start_time for disruption in disruption_list])
            self.end_time = max([disruption.start_time + disruption.duration for disruption in disruption_list])
            self.transport_nodes = [
                disruption.item_id
                for disruption in disruption_list
                if disruption.item_type == "transport_node"
            ]
            self.transport_edges = [
                disruption.item_id
                for disruption in disruption_list
                if disruption.item_type == "transport_edge"
            ]
            self.firms = [
                disruption.item_id
                for disruption in disruption_list
                if disruption.item_type == "firm"
            ]
        else:
            self.start_time = 0
            self.end_time = 0

    @classmethod
    def from_disruption_description(
            cls,
            disruption_description: dict,
            edges: geopandas.GeoDataFrame,
            firm_table: pandas.DataFrame
    ):
        disruption_list = Disruption.from_disruption_description(disruption_description, edges, firm_table)
        return cls(disruption_list)

    def print_info(self):
        print(f'There are {len(self)} disruptions')
        for disruption in self:
            disruption.print_info()

    def filter_type(self, selected_item_type):
        return DisruptionList([disruption for disruption in self if disruption.item_type == selected_item_type])

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
