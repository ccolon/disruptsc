import gc
import logging
from typing import TYPE_CHECKING, Type, List
from .base_executor import SimulationExecutor

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class MonteCarloExecutor(SimulationExecutor):
    """Executes Monte Carlo simulations using a base executor."""
    
    def __init__(self, model, parameters, base_executor_class: Type[SimulationExecutor], results_writer=None):
        super().__init__(model, parameters)
        self.base_executor_class = base_executor_class
        self.results_writer = results_writer
    
    def execute(self) -> List["Simulation"]:
        """Execute Monte Carlo iterations and return list of simulations."""
        results = []
        
        logging.info(f"{self.parameters.mc_repetitions} Monte Carlo simulations")
        
        for i in range(self.parameters.mc_repetitions):
            logging.info(f"")
            logging.info(f"=============== Starting repetition #{i} ===============")

            if i == 0:
                self._reset_model_state(full_reset=True)
            else:
                # Reset model state for each iteration
                self._reset_model_state()

            # Execute base simulation
            executor = self.base_executor_class(self.model, self.parameters)
            simulation = executor.execute()
            
            # Write iteration results if writer provided
            if self.results_writer:
                self.results_writer.write_iteration_results(i, simulation, self.model)
            
            # Clear simulation from memory immediately after processing
            del simulation
            del executor
            gc.collect()
            
            # Don't accumulate simulation objects to prevent memory leaks
            # results.append(simulation)
        
        return results
    
    def _reset_model_state(self, full_reset: bool = False):
        """Reset model state for each Monte Carlo iteration using configured caching."""
        # Use the mc_caching configuration from parameters
        if full_reset:
            self.model.setup_transport_network(False, self.parameters.with_transport)
            self.model.setup_agents(False)
            self.model.setup_sc_network(False)
            self.model.set_initial_conditions()
            self.model.setup_logistic_routes(False)
        else:
            caching_config = self.parameters.mc_caching
            self.model.setup_transport_network(caching_config['transport_network'], self.parameters.with_transport)
            self.model.setup_agents(caching_config['agents'])
            self.model.setup_sc_network(caching_config['sc_network'])
            self.model.set_initial_conditions()
            self.model.setup_logistic_routes(caching_config['logistic_routes'])


class InitialStateMCExecutor(SimulationExecutor):
    """Specialized Monte Carlo executor for initial state simulations."""
    
    def execute(self) -> List["Simulation"]:
        """Execute initial state Monte Carlo with flow aggregation."""
        import pandas as pd
        from disruptsc.model.utils.caching import load_cached_model
        
        flow_dfs = {}
        
        # Setup and save initial model state
        self.model.setup_transport_network(cached=False)
        self.model.setup_agents(cached=False)
        self.model.save_pickle('initial_state_mc')
        
        for i in range(self.parameters.mc_repetitions):
            logging.info(f"")
            logging.info(f"=============== Starting repetition #{i} ===============")
            
            # Load cached model and reset stochastic components
            model = load_cached_model("initial_state_mc")
            model.shuffle_logistic_costs()
            model.setup_sc_network(cached=False)
            model.set_initial_conditions()
            model.setup_logistic_routes(cached=False)
            
            # Run simulation
            simulation = model.run_static()
            
            # Process flow data
            flow_df = pd.DataFrame(simulation.transport_network_data)
            flow_df = flow_df[(flow_df['flow_total'] > 0) & (flow_df['time_step'] == 0)]
            
            # Save flow data to disk immediately
            flow_df.to_csv(self.parameters.export_folder / f"flow_df_{i}.csv")
            
            # Store only minimal data for aggregation
            flow_dfs[i] = flow_df[['id', 'flow_total']].copy()  # Keep only essential columns
            
            # Clear simulation and model from memory immediately
            del simulation
            del model
            del flow_df
            gc.collect()
        
        # Aggregate results
        self._aggregate_flow_results(flow_dfs)
        
        # Clear flow data from memory after aggregation
        del flow_dfs
        gc.collect()
        
        return []  # Return empty list - results are written to disk
    
    def _aggregate_flow_results(self, flow_dfs):
        """Aggregate flow results across Monte Carlo iterations."""
        import pandas as pd
        
        mean_flows = pd.concat(flow_dfs.values())
        mean_flows = mean_flows.groupby(mean_flows.index).mean()
        transport_edges_with_flows = pd.merge(
            self.model.transport_edges.drop(columns=["node_tuple"]),
            mean_flows, how="left", on="id"
        )
        transport_edges_with_flows.to_file(
            self.parameters.export_folder / f"transport_edges_with_flows.geojson",
            driver="GeoJSON", index=False
        )