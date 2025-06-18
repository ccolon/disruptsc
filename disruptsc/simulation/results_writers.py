import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from disruptsc import paths

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class CSVResultsWriter:
    """Handles writing simulation results to CSV files."""
    
    def __init__(self, output_file: Path, headers: list):
        self.output_file = output_file
        self.headers = headers
        self._write_headers()
    
    def _write_headers(self):
        """Write CSV headers."""
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)
    
    def write_row(self, data: list):
        """Write a single row of data."""
        with open(self.output_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)
    
    @classmethod
    def create_disruption_mc_writer(cls, parameters: "Parameters") -> "DisruptionMCWriter":
        """Create writer for disruption Monte Carlo results."""
        suffix = round(datetime.now().timestamp() * 1000)
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"disruption_{suffix}.csv"
        return DisruptionMCWriter(output_file, parameters)
    
    @classmethod
    def create_criticality_writer(cls, parameters: "Parameters") -> "CriticalityWriter":
        """Create writer for criticality analysis results."""
        suffix = round(datetime.now().timestamp() * 1000)
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"criticality_{suffix}.csv"
        return CriticalityWriter(output_file, parameters)
    
    @classmethod
    def create_ad_hoc_writer(cls, parameters: "Parameters") -> "AdHocWriter":
        """Create writer for ad-hoc analysis results."""
        suffix = round(datetime.now().timestamp() * 1000)
        output_file = paths.OUTPUT_FOLDER / parameters.scope / f"disruption_{suffix}.csv"
        return AdHocWriter(output_file, parameters)


class DisruptionMCWriter(CSVResultsWriter):
    """Writer for disruption Monte Carlo simulation results."""
    
    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Need to create a dummy model to get region info - this is a limitation of current design
        from disruptsc.model.model import Model
        temp_model = Model(parameters)
        temp_model._prepare_mrio_and_sectors()
        
        region_household_loss_labels = ['household_loss_' + region for region in temp_model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in temp_model.mrio.external_buying_countries]
        
        headers = ["mc_repetition", "duration", "household_loss", "country_loss"] + \
                 region_household_loss_labels + country_loss_labels
        
        super().__init__(output_file, headers)
        self.parameters = parameters
    
    def write_iteration_results(self, iteration: int, simulation: "Simulation", model: "Model"):
        """Write results for a single Monte Carlo iteration."""
        household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
        household_loss = sum(household_loss_per_region.values())
        country_loss_per_country = simulation.calculate_country_loss(per_country=True)
        country_loss = sum(country_loss_per_country.values())
        
        logging.info(f"Simulation terminated. "
                    f"Household loss: {int(household_loss)} {self.parameters.monetary_units_in_model}. "
                    f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")
        
        household_loss_per_region_values = [
            household_loss_per_region.get(region, 0.0) for region in model.mrio.regions
        ]
        country_loss_per_region_values = [
            country_loss_per_country.get(country, 0.0) for country in model.mrio.external_buying_countries
        ]
        
        self.write_row([iteration, household_loss, country_loss] + 
                      household_loss_per_region_values + country_loss_per_region_values)


class CriticalityWriter(CSVResultsWriter):
    """Writer for criticality analysis results."""
    
    def __init__(self, output_file: Path, parameters: "Parameters"):
        # Need to create a dummy model to get region info
        from disruptsc.model.model import Model
        temp_model = Model(parameters)
        temp_model._prepare_mrio_and_sectors()
        
        region_household_loss_labels = ['household_loss_' + region for region in temp_model.mrio.regions]
        country_loss_labels = ['country_loss_' + country for country in temp_model.mrio.external_buying_countries]
        
        headers = ["edge", "name", "type", parameters.criticality['attribute'], "duration",
                  "household_loss", "country_loss"] + \
                 region_household_loss_labels + country_loss_labels + ['geometry']
        
        super().__init__(output_file, headers)
        self.parameters = parameters
    
    def write_criticality_results(self, edge: int, attribute, simulation: "Simulation", model: "Model", 
                                 household_loss: float, country_loss: float):
        """Write results for a single edge criticality analysis."""
        household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
        country_loss_per_country = simulation.calculate_country_loss(per_country=True)
        
        household_loss_per_region_values = [
            household_loss_per_region.get(region, 0.0) for region in model.mrio.regions
        ]
        country_loss_per_region_values = [
            country_loss_per_country.get(country, 0.0) for country in model.mrio.external_buying_countries
        ]
        
        geometry = model.transport_edges.loc[edge, 'geometry']
        name = model.transport_edges.loc[edge, 'name']
        transport_type = model.transport_edges.loc[edge, 'type']
        
        self.write_row([edge, name, transport_type, attribute, self.parameters.criticality['duration'],
                       household_loss, country_loss] +
                      household_loss_per_region_values + country_loss_per_region_values + [geometry])


class AdHocWriter(CSVResultsWriter):
    """Writer for ad-hoc analysis results."""
    
    def __init__(self, output_file: Path, parameters: "Parameters"):
        periods = [30, 90, 180]
        household_loss_labels = ['household_loss_' + str(period) for period in periods]
        headers = ["sectors", "duration", "household_loss"] + household_loss_labels + ["country_loss"]
        
        super().__init__(output_file, headers)
        self.parameters = parameters
    
    def write_ad_hoc_results(self, disrupted_sectors: list, simulation: "Simulation", model: "Model",
                            household_loss: float, country_loss: float, household_loss_per_periods: dict):
        """Write results for a single sector combination analysis."""
        self.write_row(["_".join(disrupted_sectors), household_loss, country_loss] +
                      list(household_loss_per_periods.values()) + [country_loss])