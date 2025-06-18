from .base_executor import SimulationExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class DisruptionExecutor(SimulationExecutor):
    """Executes single disruption simulation."""
    
    def execute(self) -> "Simulation":
        return self.model.run_disruption(t_final=self.parameters.t_final)