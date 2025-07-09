import logging
from typing import TYPE_CHECKING, List
from .base_executor import SimulationExecutor

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class CriticalityExecutor(SimulationExecutor):
    """Executes criticality analysis simulations."""
    
    def __init__(self, model, parameters, results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
    
    def execute(self) -> List["Simulation"]:
        """Execute criticality analysis and return list of simulations."""
        from datetime import datetime
        from disruptsc.model.utils.caching import load_cached_model
        
        # Save model state
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)
        
        # Determine edges to test
        edges_to_test = self._get_edges_to_test()
        disruption_duration = self.parameters.criticality['duration']
        
        logging.info(f"")
        logging.info(f"")
        logging.info(f"========== Criticality simulation of {len(edges_to_test)} edges ==========")
        
        results = []
        for edge, attribute in edges_to_test.items():
            logging.info(f"")
            logging.info(f"=== Edge {edge} ====")
            
            # Load fresh model state
            model = load_cached_model(suffix)
            
            # Run criticality simulation
            simulation = model.run_criticality_disruption(edge, disruption_duration)
            
            # Calculate losses
            household_loss_per_region = simulation.calculate_household_loss(model.household_table, per_region=True)
            household_loss = sum(household_loss_per_region.values())
            country_loss_per_country = simulation.calculate_country_loss(per_country=True)
            country_loss = sum(country_loss_per_country.values())
            
            logging.info(f"Simulation terminated. "
                        f"Household loss: {int(household_loss)} {self.parameters.monetary_units_in_model}. "
                        f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")
            
            # Write results if writer provided
            if self.results_writer:
                self.results_writer.write_criticality_results(
                    edge, attribute, simulation, model, household_loss, country_loss
                )
            
            results.append(simulation)
        
        return results
    
    def _get_edges_to_test(self) -> dict:
        """Determine which edges to test based on parameters."""
        import pandas as pd
        
        if self.parameters.criticality['attribute'] == "top_flow":
            return self._get_top_flow_edges()
        elif self.parameters.criticality['attribute'] == "id":
            edges = self.parameters.criticality['edges']
            return {i: i for i in edges}
        else:
            condition = self.model.transport_edges[self.parameters.criticality['attribute']].isin(
                self.parameters.criticality['edges']
            )
            edges_to_test = self.model.transport_edges.sort_values('id')[condition]
            return edges_to_test.set_index('id')[self.parameters.criticality['attribute']].to_dict()
    
    def _get_top_flow_edges(self) -> dict:
        """Get edges based on top flow analysis."""
        import pandas as pd
        
        # Run initial simulation to get flows
        simulation = self.model.run_static()
        flow_df = pd.DataFrame(simulation.transport_network_data)
        flows = pd.merge(
            self.model.transport_edges, 
            flow_df[flow_df['time_step'] == 0],
            how="left", on="id"
        )
        flows = flows[flows['flow_total'] > 0]
        flows = flows[flows['type'].isin(['roads', 'railways'])]
        flows = flows.sort_values(by='flow_total', ascending=False)
        total = flows['flow_total'].sum()
        flows['cumulative_share'] = flows['flow_total'].cumsum() / total
        top_df = flows[flows['cumulative_share'] <= 0.9]
        return top_df.set_index('id')['flow_total'].to_dict()