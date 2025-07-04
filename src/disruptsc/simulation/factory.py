from typing import TYPE_CHECKING
from .executors.base_executor import SimulationExecutor
from .executors.initial_state_executor import InitialStateExecutor, StationaryTestExecutor, FlowCalibrationExecutor
from .executors.disruption_executor import DisruptionExecutor
from .executors.monte_carlo_executor import MonteCarloExecutor, InitialStateMCExecutor
from .executors.criticality_executor import CriticalityExecutor
from .executors.ad_hoc_executor import AdHocExecutor
from .results_writers import CSVResultsWriter

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters


class ExecutorFactory:
    """Factory for creating simulation executors based on simulation type."""
    
    @classmethod
    def create_executor(cls, simulation_type: str, model: "Model", parameters: "Parameters") -> SimulationExecutor:
        """Create appropriate executor for the given simulation type."""
        
        # Determine if this should be a Monte Carlo run
        is_monte_carlo = parameters.mc_repetitions and parameters.mc_repetitions >= 1
        
        # Base simulation types
        if simulation_type == "initial_state":
            if is_monte_carlo:
                return InitialStateMCExecutor(model, parameters, InitialStateExecutor)
            else:
                return InitialStateExecutor(model, parameters)
        
        elif simulation_type == "stationary_test":
            return StationaryTestExecutor(model, parameters)
        
        elif simulation_type in ["event", "disruption"]:
            if is_monte_carlo:
                results_writer = CSVResultsWriter.create_disruption_mc_writer(parameters)
                return MonteCarloExecutor(model, parameters, DisruptionExecutor, results_writer)
            else:
                return DisruptionExecutor(model, parameters)
        
        elif simulation_type == "flow_calibration":
            return FlowCalibrationExecutor(model, parameters)
        
        # Complex simulation types
        elif simulation_type == "criticality":
            results_writer = CSVResultsWriter.create_criticality_writer(parameters)
            return CriticalityExecutor(model, parameters, results_writer)
        
        elif simulation_type == "ad_hoc":
            results_writer = CSVResultsWriter.create_ad_hoc_writer(parameters)
            return AdHocExecutor(model, parameters, disruption_type="sectors", results_writer=results_writer)

        elif simulation_type == "ad_hoc_province":
            results_writer = CSVResultsWriter.create_ad_hoc_writer(parameters)
            return AdHocExecutor(model, parameters, disruption_type="subregions", subregion="province", results_writer=results_writer)

        elif simulation_type == "ad_hoc_canton":
            results_writer = CSVResultsWriter.create_ad_hoc_writer(parameters)
            return AdHocExecutor(model, parameters, disruption_type="subregions", subregion="canton", results_writer=results_writer)
        
        else:
            raise ValueError(f'Unimplemented simulation type: {simulation_type}')