from .base_executor import SimulationExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class InitialStateExecutor(SimulationExecutor):
    """Executes initial state (static equilibrium) simulation."""
    
    def execute(self) -> "Simulation":
        return self.model.run_static()


class StationaryTestExecutor(SimulationExecutor):
    """Executes stationary test simulation."""
    
    def execute(self) -> "Simulation":
        return self.model.run_stationary_test()


class FlowCalibrationExecutor(SimulationExecutor):
    """Executes flow calibration simulation."""
    
    def execute(self) -> "Simulation":
        from disruptsc.model.utils.functions import mean_squared_distance
        
        simulation = self.model.run_static()
        calibration_flows = simulation.report_annual_flow_specific_edges(
            self.parameters.flow_data, 
            self.model.transport_edges,
            self.parameters.time_resolution, 
            usd_or_ton='ton'
        )
        print(calibration_flows)
        print(mean_squared_distance(calibration_flows, self.parameters.flow_data))
        return simulation