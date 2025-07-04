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
            
            # Full model setup - no caching
            model.setup_transport_network(cached=False)
            model.setup_agents(cached=False)
            model.setup_sc_network(cached=False)
            model.set_initial_conditions()
            model.setup_logistic_routes(cached=False)
            
            # Run disruption simulation
            simulation = model.run_disruption(t_final=model.parameters.t_final)

            # Calculate final losses only
            household_loss = simulation.calculate_household_loss(model.household_table, periods=[180])[180]
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
            self._set_nested_parameter(modified_params, param_path, value)
        
        return modified_params
    
    def _set_nested_parameter(self, params, param_path: str, value):
        """Set nested parameter using dot notation (e.g., 'inventory_duration_targets.values.transport')."""
        parts = param_path.split('.')
        current = params
        
        # Navigate to the parent object
        for i, part in enumerate(parts[:-1]):
            if hasattr(current, part):
                attr = getattr(current, part)
                # Check if it's a method/function - if so, skip this navigation approach
                if callable(attr) and not isinstance(attr, (dict, list)):
                    # This means we hit a method like 'values()' - the structure is different than expected
                    # Fall back to direct parameter modification
                    return self._set_parameter_direct(params, param_path, value)
                current = attr
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                # Try direct parameter modification as fallback
                return self._set_parameter_direct(params, param_path, value)
        
        # Set the final value
        final_key = parts[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        elif isinstance(current, dict):
            current[final_key] = value
        else:
            # Try direct parameter modification as fallback
            return self._set_parameter_direct(params, param_path, value)
    
    def _set_parameter_direct(self, params, param_path: str, value):
        """Fallback method to set parameters directly on the Parameters object."""
        # For inventory_duration_targets.values.X, we need to modify the dict directly
        if param_path.startswith('inventory_duration_targets.values.'):
            sector_key = param_path.split('.')[-1]  # Get the final part (e.g., 'utility')
            
            # Ensure inventory_duration_targets exists and has proper structure
            if not hasattr(params, 'inventory_duration_targets'):
                logging.error(f"Parameters object has no 'inventory_duration_targets' attribute")
                raise ValueError(f"Parameter '{param_path}' not supported - no inventory_duration_targets")
            
            idt = params.inventory_duration_targets
            
            # Ensure values dict exists
            if isinstance(idt, dict):
                if 'values' not in idt:
                    idt['values'] = {}
                idt['values'][sector_key] = value
            else:
                # Handle case where idt is an object with a values attribute
                if not hasattr(idt, 'values'):
                    logging.error(f"inventory_duration_targets has no 'values' attribute: {type(idt)}")
                    raise ValueError(f"Parameter '{param_path}' not supported - inventory_duration_targets.values not found")
                
                values = idt.values
                if isinstance(values, dict):
                    values[sector_key] = value
                else:
                    logging.error(f"inventory_duration_targets.values is not a dict: {type(values)}")
                    raise ValueError(f"Parameter '{param_path}' not supported - values is not a dict")
                    
            logging.info(f"Set {param_path} = {value} via direct modification")
            return
        
        # Handle other parameter types as needed
        if '.' in param_path:
            # For other nested parameters, try to set them as simple attributes
            parts = param_path.split('.')
            if len(parts) == 2 and hasattr(params, parts[0]):
                parent = getattr(params, parts[0])
                if isinstance(parent, dict):
                    parent[parts[1]] = value
                    logging.info(f"Set {param_path} = {value} via two-level dict")
                    return
                elif hasattr(parent, parts[1]):
                    setattr(parent, parts[1], value) 
                    logging.info(f"Set {param_path} = {value} via two-level attribute")
                    return
        
        # Simple parameter - set directly on params object
        if hasattr(params, param_path):
            setattr(params, param_path, value)
            logging.info(f"Set {param_path} = {value} via direct attribute")
            return
            
        logging.error(f"Could not set parameter {param_path} = {value}")
        raise ValueError(f"Parameter '{param_path}' not supported by direct modification")
