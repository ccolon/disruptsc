from typing import TYPE_CHECKING, Dict, Any, Type
from abc import ABC, abstractmethod

import logging
from collections import UserList
from pathlib import Path

import numpy as np
import geopandas
import pandas as pd

from disruptsc.parameters import EPSILON, import_code

if TYPE_CHECKING:
    from disruptsc.agents.firm import Firms
    from disruptsc.model.model import Model
    from disruptsc.network.transport_network import TransportNetwork


class DisruptionContext:
    """Context object containing shared data for disruption creation."""
    
    def __init__(self, model_unit: str, edges: geopandas.GeoDataFrame, 
                 firm_table: pd.DataFrame, firm_list: "Firms"):
        self.model_unit = model_unit
        self.edges = edges
        self.firm_table = firm_table
        self.firm_list = firm_list


class DisruptionFactory:
    """Factory for creating disruption objects from configuration."""
    
    _disruption_types: Dict[str, Type["BaseDisruption"]] = {}
    _config_handlers: Dict[str, callable] = {}
    
    @classmethod
    def register_disruption_type(cls, disruption_type: str, disruption_class: Type["BaseDisruption"]):
        """Register a disruption type with its corresponding class."""
        cls._disruption_types[disruption_type] = disruption_class
    
    @classmethod
    def register_config_handler(cls, disruption_type: str, handler: callable):
        """Register a configuration handler for a disruption type."""
        cls._config_handlers[disruption_type] = handler
    
    @classmethod
    def create_disruption(cls, config: Dict[str, Any], context: DisruptionContext) -> "BaseDisruption":
        """Create a disruption from configuration."""
        disruption_type = config.get('type')
        if not disruption_type:
            raise ValueError("Disruption configuration must include 'type'")
        
        if disruption_type not in cls._config_handlers:
            raise ValueError(f"Unknown disruption type: {disruption_type}")
        
        handler = cls._config_handlers[disruption_type]
        return handler(config, context)
    
    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported disruption types."""
        return list(cls._disruption_types.keys())
    
    @classmethod
    def create_disruptions_from_list(cls, configs: list, context: DisruptionContext) -> list:
        """Create multiple disruptions from a list of configurations."""
        disruptions = []
        for config in configs:
            result = cls.create_disruption(config, context)
            if isinstance(result, list):
                disruptions.extend(result)
            else:
                disruptions.append(result)
        return disruptions


def _create_capital_destruction(config: Dict[str, Any], context: DisruptionContext) -> "CapitalDestruction":
    """Create capital destruction disruption from config."""
    description_type = config.get('description_type')
    
    if description_type == "region_sector_file":
        disruption = CapitalDestruction.from_region_sector_file(
            config['region_sector_filepath'],
            context.firm_table,
            context.firm_list,
            input_units=config['unit'],
            target_units=context.model_unit
        )
    elif description_type == "filter":
        disruption = CapitalDestruction.from_firms_attributes(
            config['destroyed_capital'],
            config['filter'],
            context.firm_list,
            config['unit'],
            context.model_unit
        )
    else:
        raise ValueError(f"Unknown description_type for capital_destruction: {description_type}")
    
    # Set common attributes
    disruption.start_time = config["start_time"]
    if "reconstruction_market" in config:
        disruption.reconstruction_market = config["reconstruction_market"]
    if "reconstruction_target_time" in config:
        disruption.reconstruction_target_time = config["reconstruction_target_time"]
    if "capital_input_mix" in config:
        disruption.capital_input_mix = config["capital_input_mix"]
    
    return disruption


def _create_productivity_shock(config: Dict[str, Any], context: DisruptionContext) -> "ProductivityShock":
    """Create productivity shock disruption from config."""
    description_type = config.get('description_type')
    
    if description_type == "region_sector_file":
        disruption = ProductivityShock.from_region_sector_file(
            config['region_sector_filepath'],
            context.firm_table,
            context.firm_list
        )
    elif description_type == "filter":
        disruption = ProductivityShock.from_firms_attributes(
            config['productivity_reduction'],
            config['filter'],
            context.firm_list
        )
    else:
        raise ValueError(f"Unknown description_type for productivity_shock: {description_type}")
    
    # Set common attributes
    disruption.start_time = config["start_time"]
    if "duration" in config:
        shape = config.get("recovery_shape", "threshold")
        rate = config.get("recovery_rate", 1.0)
        disruption.recovery = Recovery(duration=config["duration"], shape=shape, rate=rate)
    
    return disruption


def _create_transport_disruption(config: Dict[str, Any], context: DisruptionContext) -> "TransportDisruption":
    """Create transport disruption from config."""
    description_type = config.get('description_type')
    
    if description_type == "edge_attributes":
        disruption = TransportDisruption.from_edge_attributes(
            edges=context.edges,
            attribute=config['attribute'],
            values=config['values']
        )
    else:
        raise ValueError(f"Unknown description_type for transport_disruption: {description_type}")
    
    # Set common attributes
    disruption.start_time = config["start_time"]
    if "duration" in config:
        shape = config.get("recovery_shape", "threshold")
        rate = config.get("recovery_rate", 1.0)
        disruption.recovery = Recovery(duration=config["duration"], shape=shape, rate=rate)
    
    return disruption


def _create_transport_disruption_probability(config: Dict[str, Any], context: DisruptionContext) -> list:
    """Create probabilistic transport disruptions from config."""
    description_type = config.get('description_type')
    
    if description_type == "edge_attributes":
        base_disruption = TransportDisruption.from_edge_attributes(
            edges=context.edges,
            attribute=config['attribute'],
            values=config['values']
        )
        
        start_times, durations = DisruptionList.generate_disruption_scenario_from_probabilities(
            config["scenario_duration"],
            probability_duration_pairs=config['probability_duration_pairs']
        )
        
        disruptions = []
        for start_time, duration in zip(start_times, durations):
            shape = config.get("recovery_shape", "threshold")
            rate = config.get("recovery_rate", 1.0)
            disruption = TransportDisruption(
                description=base_disruption.description,
                recovery=Recovery(duration=duration, shape=shape, rate=rate),
                start_time=start_time
            )
            disruptions.append(disruption)
        
        return disruptions
    else:
        raise ValueError(f"Unknown description_type for transport_disruption_probability: {description_type}")


# Register disruption types and handlers
DisruptionFactory.register_disruption_type("capital_destruction", "CapitalDestruction")
DisruptionFactory.register_disruption_type("productivity_shock", "ProductivityShock") 
DisruptionFactory.register_disruption_type("transport_disruption", "TransportDisruption")
DisruptionFactory.register_disruption_type("transport_disruption_probability", "TransportDisruption")

DisruptionFactory.register_config_handler("capital_destruction", _create_capital_destruction)
DisruptionFactory.register_config_handler("productivity_shock", _create_productivity_shock)
DisruptionFactory.register_config_handler("transport_disruption", _create_transport_disruption)
DisruptionFactory.register_config_handler("transport_disruption_probability", _create_transport_disruption_probability)


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
    def __init__(self, duration: int, shape: str = "linear", rate: float = 1.0):
        self.duration = duration
        self.shape = shape  # "linear", "exponential", "threshold"
        self.rate = rate  # Recovery rate parameter
    
    def get_recovery_factor(self, time_since_start: int) -> float:
        """Calculate recovery factor based on time since disruption start."""
        if time_since_start >= self.duration:
            return 1.0
        
        progress = time_since_start / self.duration
        
        if self.shape == "threshold":
            return 0.0 if time_since_start < self.duration else 1.0
        elif self.shape == "linear":
            return progress * self.rate
        elif self.shape == "exponential":
            return (1 - np.exp(-self.rate * progress)) / (1 - np.exp(-self.rate))
        else:
            raise ValueError(f"Unknown recovery shape: {self.shape}")


class BaseDisruption(ABC):
    """Abstract base class for all disruption types."""
    
    def __init__(self, description: dict, recovery: Recovery = None, start_time: int = 1):
        self.start_time = start_time
        self.recovery = recovery
        self.description = description or {}
        self._validate_description()
    
    @abstractmethod
    def _validate_description(self):
        """Validate the description dictionary for this disruption type."""
        pass
    
    @abstractmethod
    def implement(self, model: "Model"):
        """Implement the disruption in the model."""
        pass
    
    def log_info(self):
        """Log information about this disruption."""
        recovery_info = f"with {self.recovery.shape} recovery over {self.recovery.duration} steps" if self.recovery else "with no recovery"
        logging.info(f"{self.__class__.__name__}: {len(self.description)} items disrupted at time {self.start_time} {recovery_info}")
    
    def __len__(self):
        return len(self.description)
    
    def keys(self):
        return self.description.keys()
    
    def items(self):
        return self.description.items()
    
    def values(self):
        return self.description.values()
    
    def __getitem__(self, key):
        return self.description[key]
    
    def __contains__(self, key):
        return key in self.description


class TransportDisruption(BaseDisruption):
    def __init__(self, description: dict, recovery: Recovery = None, start_time: int = 1):
        super().__init__(description, recovery, start_time)

    def _validate_description(self):
        """Validate transport disruption description."""
        for key, value in self.description.items():
            if not isinstance(key, int):
                raise KeyError("Key must be an int: the id of the transport edge to be disrupted")
            if not isinstance(value, float):
                raise ValueError("Value must be a float: the fraction of lost capacity")
            if not 0 <= value <= 1:
                raise ValueError("Capacity loss fraction must be between 0 and 1")

    def __repr__(self):
        return f"TransportDisruption(start_time={self.start_time}, edges={len(self.description)}, recovery={self.recovery})"

    @classmethod
    def from_edge_attributes(cls, edges: geopandas.GeoDataFrame, attribute: str, values: list):
        # we do a special case for the disruption attribute
        # for which we check if the attribute contains one of the value
        if attribute == "disruption":
            condition = [edges[attribute].str.contains(value) for value in values]
            condition = pd.concat(condition, axis=1)
            condition = condition.any(axis=1)
        else:
            condition = edges[attribute].isin(values)
        item_ids = edges.sort_values('id').loc[condition, 'id'].tolist()
        description = pd.Series(1.0, index=item_ids).to_dict()

        return cls(description=description)

    def implement(self, transport_network: "TransportNetwork"):
        """Implement transport disruption."""
        for edge in transport_network.edges:
            edge_id = transport_network[edge[0]][edge[1]]['id']
            if edge_id in self.keys():
                duration = self.recovery.duration if self.recovery else float('inf')
                transport_network.disrupt_one_edge(edge, self[edge_id], duration)


class CapitalDestruction(BaseDisruption):
    def __init__(self, description: dict, recovery: Recovery = None, start_time: int = 1,
                 reconstruction_market: bool = False, reconstruction_target_time: int = 30,
                 capital_input_mix: dict = None):
        self.reconstruction_market = reconstruction_market
        self.reconstruction_target_time = reconstruction_target_time
        self.capital_input_mix = capital_input_mix or {"CON": 0.7, "MAN": 0.2, "IMP": 0.1}
        super().__init__(description, recovery, start_time)

    def _validate_description(self):
        """Validate capital destruction description."""
        for key, value in self.description.items():
            if not isinstance(key, int):
                raise KeyError("Key must be an int: the id of the firm")
            if not isinstance(value, (int, float)):
                raise ValueError("Value must be a number: the amount of destroyed capital")
            if value < 0:
                raise ValueError("Destroyed capital must be non-negative")

    def __repr__(self):
        return f"CapitalDestruction(start_time={self.start_time}, firms={len(self.description)}, reconstruction={self.reconstruction_market})"

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

        return cls(description=result, recovery=None)

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
        return cls(description=description, recovery=None)

    def implement(self, model: "Model"):
        """Implement capital destruction."""
        firms = model.firms
        for firm_id, destroyed_capital in self.items():
            firms[firm_id].incur_capital_destruction(destroyed_capital)
        if self.reconstruction_market:
            model.reconstruction_market = ReconstructionMarket(
                reconstruction_target_time=self.reconstruction_target_time,
                capital_input_mix=self.capital_input_mix
            )


class ProductivityShock(BaseDisruption):
    """Disruption affecting firm productivity for a specified duration."""
    
    def __init__(self, description: dict, recovery: Recovery = None, start_time: int = 1):
        super().__init__(description, recovery, start_time)
    
    def _validate_description(self):
        """Validate productivity shock description."""
        for key, value in self.description.items():
            if not isinstance(key, int):
                raise KeyError("Key must be an int: the id of the firm")
            if not isinstance(value, (int, float)):
                raise ValueError("Value must be a number: the productivity reduction factor")
            if not 0 <= value <= 1:
                raise ValueError("Productivity reduction factor must be between 0 and 1")
    
    def __repr__(self):
        return f"ProductivityShock(start_time={self.start_time}, firms={len(self.description)}, recovery={self.recovery})"
    
    @classmethod
    def from_firms_attributes(cls, productivity_reduction: float, filters: dict, firms: "Firms"):
        """Create productivity shock from firm attributes and filters."""
        if not 0 <= productivity_reduction <= 1:
            raise ValueError("Productivity reduction must be between 0 and 1")
        
        affected_firms = firms.select_by_properties(filters)
        description = {firm_id: productivity_reduction for firm_id in affected_firms.keys()}
        
        return cls(description=description)
    
    @classmethod
    def from_region_sector_file(cls, filepath: Path, firm_table: pd.DataFrame, firms: "Firms"):
        """Create productivity shock from region-sector file."""
        df = pd.read_csv(filepath, dtype={'region': str, 'sector': str, 'productivity_reduction': float})
        result = {}
        
        for _, row in df.iterrows():
            region, sector, reduction = row['region'], row['sector'], row['productivity_reduction']
            firm_ids = firm_table.loc[
                (firm_table['region'] == region) & (firm_table['sector'] == sector), "id"
            ].tolist()
            
            if len(firm_ids) == 0:
                logging.warning(f"In region {region}, sector {sector}: no firms found for productivity shock")
            else:
                for firm_id in firm_ids:
                    result[firm_id] = reduction
        
        return cls(description=result)
    
    def implement(self, model: "Model"):
        """Implement productivity shock."""
        firms = model.firms
        for firm_id, reduction_factor in self.items():
            if hasattr(firms[firm_id], 'apply_productivity_shock'):
                firms[firm_id].apply_productivity_shock(reduction_factor, self.recovery)
            else:
                logging.warning(f"Firm {firm_id} does not support productivity shocks")


class DisruptionList(UserList):
    def __init__(self, disruption_list: list):
        super().__init__(disruption for disruption in disruption_list
                         if isinstance(disruption, BaseDisruption))
        if len(disruption_list) > 0:
            self.start_time = min([disruption.start_time for disruption in disruption_list])
            self.end_time = max([disruption.start_time + 1 for disruption in disruption_list])
        else:
            self.start_time = 0
            self.end_time = 0

    @classmethod
    def from_disruptions_parameter(
            cls,
            disruptions: list,
            model_unit: str,
            edges: geopandas.GeoDataFrame,
            firm_table: pd.DataFrame,
            firm_list: "Firms"
    ):
        """Create DisruptionList from configuration using factory pattern."""
        context = DisruptionContext(model_unit, edges, firm_table, firm_list)
        
        try:
            disruption_list = DisruptionFactory.create_disruptions_from_list(disruptions, context)
            return cls(disruption_list)
        except Exception as e:
            logging.error(f"Failed to create disruptions from configuration: {e}")
            raise

    @classmethod
    def register_disruption_type(cls, disruption_type: str, handler: callable):
        """Register a new disruption type with the factory."""
        DisruptionFactory.register_config_handler(disruption_type, handler)
    
    @classmethod 
    def get_supported_disruption_types(cls) -> list:
        """Get list of supported disruption types."""
        return DisruptionFactory.get_supported_types()

    @staticmethod
    def generate_disruption_scenario_from_probabilities(scenario_duration_in_days: int = 365, probability_duration_pairs=None, seed=None):
        """
        Simulates disruptions over a number of days with multiple (probability, duration) pairs.
        Each day, for each pair, there is a chance of a disruption starting with that duration.
        Overlapping disruptions are merged.

        Parameters:
            scenario_duration_in_days (int): Number of days to simulate (default: 365)
            probability_duration_pairs (list of tuples): List of (probability, duration) pairs
                e.g., [(0.1, 1), (0.01, 10)] means 10% chance of 1-day disruption, 1% chance of 10-day disruption
            seed (int or None): Random seed for reproducibility

        Returns:
            disruption_times (list of int): Start days of merged disruptions
            disruption_durations (list of int): Durations of merged disruptions
            
        Example:
            With probability_duration_pairs=[(0.1, 1), (0.01, 10)]:
            Each day has 10% chance of 1-day disruption and 1% chance of 10-day disruption.
            If disruptions occur at t=3 (4 days) and t=5 (4 days):
            - First disruption: days 3,4,5,6
            - Second disruption: days 5,6,7,8  
            - Merged: one disruption at t=3 for 6 days (3,4,5,6,7,8)
        """
        if seed is not None:
            np.random.seed(seed)

        if probability_duration_pairs is None:
            raise ValueError("probability_duration_pairs is required")

        all_intervals = []

        # Generate disruptions for each probability-duration pair
        for prob, dur in probability_duration_pairs:
            # Generate disruption start days for this probability
            disruption_starts = np.random.rand(scenario_duration_in_days) < prob
            disruption_start_days = np.where(disruption_starts)[0]
            
            # Create intervals for each disruption [start, end)
            intervals = [(start, start + dur) for start in disruption_start_days]
            all_intervals.extend(intervals)

        if len(all_intervals) == 0:
            return [], []
        
        # Sort all intervals by start time
        all_intervals.sort()
        
        # Merge overlapping intervals
        merged_intervals = []
        current_start, current_end = all_intervals[0]
        
        for start, end in all_intervals[1:]:
            if start <= current_end:  # Overlapping or adjacent
                # Merge by extending the current interval
                current_end = max(current_end, end)
            else:
                # No overlap, save current interval and start new one
                merged_intervals.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add the last interval
        merged_intervals.append((current_start, current_end))
        
        # Convert back to start times and durations
        disruption_times = [start for start, end in merged_intervals]
        disruption_durations = [end - start for start, end in merged_intervals]

        return disruption_times, disruption_durations

    def log_info(self):
        logging.info(f'There are {len(self)} disruptions')
        for disruption in self:
            disruption.log_info()

    def filter_start_time(self, selected_start_time):
        return DisruptionList([disruption for disruption in self if disruption.start_time == selected_start_time])
