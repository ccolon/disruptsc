from typing import TYPE_CHECKING

import logging

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

FINAL_DEMAND_LABEL = "final_demand"
EXPORT_LABEL = "Exports"
IMPORT_LABEL = "Imports"
EPSILON = 1e-6


class Mrio(pd.DataFrame):
    _metadata = ['region_sectors', "region_sector_names", "regions", "sectors", "external_buying_countries",
                 "external_selling_countries", "region_households"]

    def __init__(self, *args, monetary_units, **kwargs):
        super(Mrio, self).__init__(*args, **kwargs)
        self.check_square_structure()
        self.region_sectors = [tup for tup in self.columns if tup[1] not in [FINAL_DEMAND_LABEL, EXPORT_LABEL]]
        self.region_sector_names = ['_'.join(tup) for tup in self.region_sectors]
        self.regions = list(set([tup[0] for tup in self.region_sectors]))
        self.sectors = list(set([tup[1] for tup in self.region_sectors]))
        self.external_buying_countries = [tup[0] for tup in self.columns if tup[1] == EXPORT_LABEL]
        self.external_selling_countries = [tup[0] for tup in self.index if tup[1] == IMPORT_LABEL]
        self.region_households = [tup for tup in self.columns.to_list() if tup[1] == FINAL_DEMAND_LABEL]
        self.monetary_units = monetary_units
        self.adjust_output()

    @classmethod
    def load_mrio_from_filepath(cls, filepath_mrio: "Path", monetary_units: str):
        table = pd.read_csv(filepath_mrio, header=[0, 1], index_col=[0, 1])
        # remove region_sectors with no flows
        zero_output = table.index[table.sum(axis=1) == 0].to_list()
        zero_input = table.columns[table.sum(axis=0) == 0].to_list()
        no_flow_region_sectors = list(set(zero_output) & set(zero_input))
        table.drop(index=no_flow_region_sectors, inplace=True)
        table.drop(columns=no_flow_region_sectors, inplace=True)
        return cls(table, monetary_units=monetary_units)

    def get_total_output_per_region_sectors(self):
        return self.loc[self.region_sectors].sum(axis=1)

    def get_total_input_per_region_sectors(self):
        return self[self.region_sectors].sum()

    def get_region_sectors_with_internal_flows(self, threshold: float = 0):
        tot_outputs = self.get_total_output_per_region_sectors()
        matrix_output = pd.concat([tot_outputs] * len(self.index), axis=1).transpose()
        matrix_output.index = self.index
        tech_coef_matrix = self[self.region_sectors] / matrix_output
        return [tup for tup in self.region_sectors if tech_coef_matrix.loc[tup, tup] > threshold]

    def get_final_demand(self, selected_region_sectors=None):
        if selected_region_sectors:
            if isinstance(selected_region_sectors[0], str):
                selected_region_sectors = [tuple(region_sector.split('_', 1))
                                           for region_sector in selected_region_sectors]
            elif isinstance(selected_region_sectors[0], tuple):
                pass
            else:
                raise ValueError('selected_region_sectors should be a list of tuples or strings')
            return self.loc[selected_region_sectors, self.columns.get_level_values(1) == FINAL_DEMAND_LABEL]
        else:
            return self.loc[:, self.columns.get_level_values(1) == FINAL_DEMAND_LABEL]

    def check_square_structure(self):
        region_sectors_in_columns = [tup for tup in self.columns if tup[1] not in [FINAL_DEMAND_LABEL, EXPORT_LABEL]]
        region_sectors_in_columns.sort()
        region_sectors_in_rows = [tup for tup in self.index if tup[1] != IMPORT_LABEL]
        region_sectors_in_rows.sort()
        if region_sectors_in_columns != region_sectors_in_rows:
            logging.info(set(region_sectors_in_columns) - set(region_sectors_in_rows))
            logging.info(set(region_sectors_in_rows) - set(region_sectors_in_columns))
            raise ValueError('Inconsistencies between scope sectors in rows and columns')

    def adjust_output(self):
        total_output = self.get_total_output_per_region_sectors()
        total_input = self.get_total_input_per_region_sectors()
        unbalanced_region_sectors = total_input[total_input > total_output].index.to_list()
        if len(unbalanced_region_sectors) > 0:
            logging.warning(f"There are {len(unbalanced_region_sectors)} region_sectors with more inputs than outputs")
            # TODO very ad-hoc and dirty to accommodate with bad input file
            for region_sector in unbalanced_region_sectors:
                self.loc[region_sector, ('ROW', EXPORT_LABEL)] += total_input[region_sector] - total_output[
                    region_sector] + EPSILON

    def get_tech_coef_dict(self, threshold=0, selected_region_sectors=None):
        """
        returns a dict region_sector = {input_region_sector_1: tech_coef, ...}
        """
        tot_outputs = self.get_total_output_per_region_sectors()
        matrix_output = pd.concat([tot_outputs] * len(self.index), axis=1).transpose()
        matrix_output.index = self.index
        tech_coef_matrix = self[self.region_sectors] / matrix_output

        if selected_region_sectors:
            if isinstance(selected_region_sectors[0], str):
                selected_region_sectors = [tuple(region_sector.split('_', 1))
                                           for region_sector in selected_region_sectors]
            elif isinstance(selected_region_sectors[0], tuple):
                pass
            else:
                raise ValueError('selected_region_sectors should be a list of tuples or strings')
            return {
                '_'.join(buying_region_sector_tuple): {
                    '_'.join(supplying_region_sector_tuple): val
                    for supplying_region_sector_tuple, val in sublist.items()
                    if (val > threshold) and (
                        (supplying_region_sector_tuple in selected_region_sectors)  # domestic supplier
                        or (supplying_region_sector_tuple[0] in self.external_selling_countries)  # country supplier
                    )
                }
                for buying_region_sector_tuple, sublist in tech_coef_matrix.to_dict().items()
                if buying_region_sector_tuple in selected_region_sectors
            }
        else:
            return {
                '_'.join(region_sector_tuple): {
                    '_'.join(key): val
                    for key, val in sublist.items()
                    if val > threshold
                }
                for region_sector_tuple, sublist in tech_coef_matrix.to_dict().items()
            }
