import gc
import logging
import itertools
from copy import deepcopy
from typing import TYPE_CHECKING, List

from .base_executor import SimulationExecutor

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class SensitivityExecutor(SimulationExecutor):
    """Executes sensitivity analysis across parameter combinations."""

    def __init__(self, model, parameters, results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer

    def execute(self) -> List["Simulation"]:
        """Execute sensitivity analysis and return empty list (results written to CSV)."""
        from disruptsc.model.model import Model

        if not self.parameters.sensitivity:
            raise ValueError("No sensitivity parameters defined")

        # Generate parameter combinations
        combinations = self._generate_parameter_combinations()
        logging.info(f"{len(combinations)} parameter combinations to test")

        for i, combination in enumerate(combinations):
            logging.info(f"")
            logging.info(f"=============== Starting combination #{i}: {combination} ===============")

            # Create fresh model with modified parameters
            modified_params = self._apply_parameter_combination(combination)
            model = Model(modified_params)
            if i == 0:
                _reset_model_state(model, modified_params, full_reset=True)
            else:
                # Reset model state for each iteration
                _reset_model_state(model, modified_params)

            # Run disruption simulation
            simulation = model.run_disruption(t_final=model.parameters.t_final)

            # Calculate final losses only
            household_loss = simulation.calculate_household_loss(model.household_table)
            country_loss = simulation.calculate_country_loss()

            logging.info(f"Combination #{i} completed. "
                         f"Household loss: {household_loss}. "
                         f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")

            # Write results if writer provided
            if self.results_writer:
                self.results_writer.write_sensitivity_results(i, combination, household_loss, country_loss)

            # Clear from memory immediately
            del simulation
            del model
            gc.collect()

        return []  # Return empty list - results are written to CSV

    def _generate_parameter_combinations(self) -> List[dict]:
        """Generate all parameter combinations using cartesian product."""
        param_names = []
        param_values = []

        for param_name, values in self.parameters.sensitivity.items():
            param_names.append(param_name)
            param_values.append(values)

        # Generate cartesian product
        combinations = []
        for combination in itertools.product(*param_values):
            combo_dict = dict(zip(param_names, combination))
            combinations.append(combo_dict)

        return combinations

    def _apply_parameter_combination(self, combination: dict):
        """Apply parameter combination to create modified parameters object."""
        # Deep copy original parameters
        modified_params = deepcopy(self.parameters)

        # Apply each parameter in the combination
        for param_path, value in combination.items():
            parts = param_path.split('.')
            if param_path == "duration":
                self._set_disruption_duration(modified_params, int(value))
            elif len(parts) == 1:
                self._set_simple_parameters(modified_params, parts[0], value)
            else:
                self._set_nested_parameter(modified_params, parts, value)

        return modified_params

    @staticmethod
    def _set_disruption_duration(params, duration: int):
        for disruption in params.disruptions:
            disruption['duration'] = duration
        return

    @staticmethod
    def _set_nested_parameter(params, parts: list, value):
        """Set nested parameter using dot notation (e.g., 'inventory_duration_targets.values.transport')."""
        # Navigate to the parent object
        print(parts)
        param_dict = getattr(params, parts[0])
        for part in parts[1:-1]:
            param_dict = param_dict[part]
        param_dict[parts[-1]] = value

    @staticmethod
    def _set_simple_parameters(params, parameter: str, value):
        if hasattr(params, parameter):
            setattr(params, parameter, value)
            logging.info(f"Set {parameter} = {value} via direct attribute")
            return


def _reset_model_state(model, params, full_reset: bool = False):
    """Reset model state for each Monte Carlo iteration using configured caching."""
    # Use the mc_caching configuration from parameters
    if full_reset:
        model.setup_transport_network(False, params.with_transport)
        model.setup_agents(False)
        model.setup_sc_network(False)
        model.set_initial_conditions()
        model.setup_logistic_routes(False)
    else:
        caching_config = params.mc_caching
        model.setup_transport_network(caching_config['transport_network'], params.with_transport)
        model.setup_agents(caching_config['agents'])
        model.setup_sc_network(caching_config['sc_network'])
        model.set_initial_conditions()
        model.setup_logistic_routes(caching_config['logistic_routes'])
