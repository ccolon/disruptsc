import logging
from typing import TYPE_CHECKING, List
from .base_executor import SimulationExecutor

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


class AdHocExecutor(SimulationExecutor):
    """Executes ad-hoc disruption analysis across multiple sectors."""
    
    def __init__(self, model, parameters, results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
    
    def execute(self) -> List["Simulation"]:
        """Execute ad-hoc analysis and return list of simulations."""
        from datetime import datetime
        from disruptsc.model.caching_functions import load_cached_model
        
        # Save model state
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)
        
        # Get disrupted sector combinations
        disrupted_sector_list = self._get_disrupted_sector_list()
        present_sectors = list(set(self.model.firms.get_properties('region_sector', 'list')))
        periods = [30, 90, 180]
        
        logging.info(f"{len(disrupted_sector_list)} sector combinations to test")
        
        results = []
        for disrupted_sectors in disrupted_sector_list:
            if disrupted_sectors == 'all':
                disrupted_sectors = present_sectors
            
            disrupted_sectors = ['ECU_' + sector for sector in disrupted_sectors]
            
            # Skip if sectors not present
            if any([sector not in present_sectors for sector in disrupted_sectors]):
                continue
            
            logging.info(f"")
            logging.info(f"=============== Disrupting sector #{disrupted_sectors} ===============")
            
            # Load fresh model state
            model = load_cached_model(suffix)
            model.parameters.events[0]['region_sectors'] = disrupted_sectors
            
            # Run simulation
            simulation = model.run_disruption(t_final=periods[-1])
            
            # Calculate losses
            household_loss_per_periods = simulation.calculate_household_loss(
                model.household_table, periods=periods
            )
            household_loss = household_loss_per_periods[periods[-1]]
            country_loss = simulation.calculate_country_loss()
            
            logging.info(f"Simulation terminated. "
                        f"Household loss: {household_loss_per_periods}. "
                        f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")
            
            # Write results if writer provided
            if self.results_writer:
                self.results_writer.write_ad_hoc_results(
                    disrupted_sectors, simulation, model, 
                    household_loss, country_loss, household_loss_per_periods
                )
            
            results.append(simulation)
        
        return results
    
    def _get_disrupted_sector_list(self) -> List:
        """Get the list of sector combinations to test."""
        # This would ideally be loaded from a config file
        # For now, keeping the original hardcoded list
        return ['all', ['ADM'], ['ADP'], ['ASO'], ['CAR'], ['CIN'], ['COM'], ['CON'], ['EDU'], ['ELE'], ['FIN'], ['FRT'], ['FRV'], ['INM'], ['MAQ'], ['MIP'], ['PES'], ['PPR'], ['QU2'], ['REF'], ['REP'], ['RES'], ['SAL'], ['SIL'], ['TEL'], ['TRA'], ('AGU', 'AYG'), ('AGU', 'BAL'), ('AGU', 'CEM'), ('AGU', 'CER'), ('AGU', 'DEM'), ('AGU', 'DOM'), ('AGU', 'GAN'), ('AGU', 'HIL'), ('AGU', 'LAC'), ('AGU', 'MAD'), ('AGU', 'MAN'), ('AGU', 'MOL'), ('AGU', 'MUE'), ('AGU', 'PAN'), ('AGU', 'PAP'), ('AGU', 'PLS'), ('AGU', 'SEG'), ('AGU', 'VES'), ('ALD', 'BAL'), ('ALD', 'CEM'), ('ALD', 'CER'), ('ALD', 'DOM'), ('ALD', 'GAN'), ('ALD', 'MAD'), ('ALD', 'MAN'), ('ALD', 'MOL'), ('ALD', 'PAP'), ('ALD', 'SEG'), ('AYG', 'BAL'), ('AYG', 'CEM'), ('AYG', 'CER'), ('AYG', 'DEM'), ('AYG', 'DOM'), ('AYG', 'GAN'), ('AYG', 'HIL'), ('AYG', 'LAC'), ('AYG', 'MAD'), ('AYG', 'MAN'), ('AYG', 'MOL'), ('AYG', 'MUE'), ('AYG', 'PAN'), ('AYG', 'PAP'), ('AYG', 'PLS'), ('AYG', 'SEG'), ('AYG', 'VES'), ('AZU', 'BAL'), ('AZU', 'CEM'), ('AZU', 'CER'), ('BAL', 'BNA'), ('BAL', 'CAN'), ('BAL', 'CAU'), ('BAL', 'CEM'), ('BAL', 'CER'), ('BAL', 'CHO'), ('BAL', 'CUE'), ('BAL', 'CUL'), ('BAL', 'DEM'), ('BAL', 'DOM'), ('BAL', 'FID'), ('BAL', 'GAN'), ('BAL', 'HIL'), ('BAL', 'HOT'), ('BAL', 'LAC'), ('BAL', 'MAD'), ('BAL', 'MAN'), ('BAL', 'MET'), ('BAL', 'MOL'), ('BAL', 'MUE'), ('BAL', 'PAN'), ('BAL', 'PAP'), ('BAL', 'PLS'), ('BAL', 'POS'), ('BAL', 'QU1'), ('BAL', 'SEG'), ('BAL', 'TAB'), ('BAL', 'VES'), ('BAL', 'VID'), ('BNA', 'CEM'), ('BNA', 'CER'), ('BNA', 'DEM'), ('BNA', 'DOM'), ('BNA', 'GAN'), ('BNA', 'MAD'), ('BNA', 'MAN'), ('BNA', 'MOL'), ('BNA', 'PAP'), ('BNA', 'SEG'), ('CAN', 'CEM'), ('CAN', 'CER'), ('CAU', 'CEM'), ('CAU', 'CER'), ('CEM', 'CER'), ('CEM', 'CHO'), ('CEM', 'CUE'), ('CEM', 'CUL'), ('CEM', 'DEM'), ('CEM', 'DOM'), ('CEM', 'FID'), ('CEM', 'GAN'), ('CEM', 'HIL'), ('CEM', 'HOT'), ('CEM', 'LAC'), ('CEM', 'MAD'), ('CEM', 'MAN'), ('CEM', 'MET'), ('CEM', 'MOL'), ('CEM', 'MUE'), ('CEM', 'PAN'), ('CEM', 'PAP'), ('CEM', 'PLS'), ('CEM', 'POS'), ('CEM', 'QU1'), ('CEM', 'SEG'), ('CEM', 'TAB'), ('CEM', 'VES'), ('CEM', 'VID'), ('CER', 'CHO'), ('CER', 'CUE'), ('CER', 'CUL'), ('CER', 'DEM'), ('CER', 'DOM'), ('CER', 'GAN'), ('CER', 'HIL'), ('CER', 'HOT'), ('CER', 'LAC'), ('CER', 'MAD'), ('CER', 'MAN'), ('CER', 'MET'), ('CER', 'MOL'), ('CER', 'MUE'), ('CER', 'PAN'), ('CER', 'PAP'), ('CER', 'PLS'), ('CER', 'POS'), ('CER', 'QU1'), ('CER', 'SEG'), ('CER', 'VES'), ('CER', 'VID'), ('CUL', 'GAN'), ('CUL', 'MAD'), ('CUL', 'MAN'), ('CUL', 'MOL'), ('CUL', 'PAP'), ('DEM', 'DOM'), ('DEM', 'GAN'), ('DEM', 'HIL'), ('DEM', 'HOT'), ('DEM', 'LAC'), ('DEM', 'MAD'), ('DEM', 'MAN'), ('DEM', 'MET'), ('DEM', 'MOL'), ('DEM', 'MUE'), ('DEM', 'PAN'), ('DEM', 'PAP'), ('DEM', 'PLS'), ('DEM', 'SEG'), ('DEM', 'VES'), ('DEM', 'VID'), ('DOM', 'GAN'), ('DOM', 'HIL'), ('DOM', 'HOT'), ('DOM', 'LAC'), ('DOM', 'MAD'), ('DOM', 'MAN'), ('DOM', 'MET'), ('DOM', 'MOL'), ('DOM', 'MUE'), ('DOM', 'PAN'), ('DOM', 'PAP'), ('DOM', 'PLS'), ('DOM', 'SEG'), ('DOM', 'VES'), ('DOM', 'VID'), ('GAN', 'HIL'), ('GAN', 'HOT'), ('GAN', 'LAC'), ('GAN', 'MAD'), ('GAN', 'MAN'), ('GAN', 'MET'), ('GAN', 'MOL'), ('GAN', 'MUE'), ('GAN', 'PAN'), ('GAN', 'PAP'), ('GAN', 'PLS'), ('GAN', 'QU1'), ('GAN', 'SEG'), ('GAN', 'VES'), ('GAN', 'VID'), ('HIL', 'MAD'), ('HIL', 'MAN'), ('HIL', 'MOL'), ('HIL', 'PAN'), ('HIL', 'PAP'), ('HIL', 'SEG'), ('HIL', 'VES'), ('HOT', 'MAD'), ('HOT', 'MAN'), ('HOT', 'MOL'), ('HOT', 'PAP'), ('HOT', 'SEG'), ('LAC', 'MAD'), ('LAC', 'MAN'), ('LAC', 'MOL'), ('LAC', 'PAN'), ('LAC', 'PAP'), ('LAC', 'SEG'), ('MAD', 'MAN'), ('MAD', 'MET'), ('MAD', 'MOL'), ('MAD', 'MUE'), ('MAD', 'PAN'), ('MAD', 'PAP'), ('MAD', 'PLS'), ('MAD', 'SEG'), ('MAD', 'VES'), ('MAD', 'VID'), ('MAN', 'MET'), ('MAN', 'MOL'), ('MAN', 'MUE'), ('MAN', 'PAN'), ('MAN', 'PAP'), ('MAN', 'PLS'), ('MAN', 'SEG'), ('MAN', 'VES'), ('MAN', 'VID'), ('MET', 'MOL'), ('MET', 'PAN'), ('MET', 'PAP'), ('MET', 'SEG'), ('MOL', 'MUE'), ('MOL', 'PAN'), ('MOL', 'PAP'), ('MOL', 'PLS'), ('MOL', 'SEG'), ('MOL', 'VES'), ('MOL', 'VID'), ('MUE', 'PAN'), ('MUE', 'PAP'), ('MUE', 'SEG'), ('MUE', 'VES'), ('PAN', 'PAP'), ('PAN', 'PLS'), ('PAN', 'SEG'), ('PAN', 'VES'), ('PAP', 'PLS'), ('PAP', 'SEG'), ('PAP', 'VES'), ('PAP', 'VID'), ('PLS', 'SEG'), ('PLS', 'VES'), ('SEG', 'VES'), ('SEG', 'VID')]