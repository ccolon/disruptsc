from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disruptsc.model.model import Model
    from disruptsc.parameters import Parameters
    from disruptsc.simulation.simulation import Simulation


class SimulationExecutor(ABC):
    """Base class for simulation execution strategies."""
    
    def __init__(self, model: "Model", parameters: "Parameters"):
        self.model = model
        self.parameters = parameters
    
    @abstractmethod
    def execute(self) -> "Simulation":
        """Execute the simulation and return a Simulation object with results."""
        pass