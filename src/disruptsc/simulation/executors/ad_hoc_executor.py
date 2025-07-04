import gc
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List
from .base_executor import SimulationExecutor

if TYPE_CHECKING:
    from disruptsc.simulation.simulation import Simulation


def _get_disrupted_sector_list() -> List:
    """Get the list of sector combinations to test."""
    # This would ideally be loaded from a config file
    # For now, keeping the original hardcoded list
    return ['all', ['ADM'], ['ADP'], ['ALD'], ['ASO'], ['AYG'], ['BAL'], ['CAR'], ['CIN'], ['COM'], ['CON'], ['DEM'], ['EDU'], ['ELE'], ['FIN'], ['FRT'], ['FRV'], ['GAN'], ['INM'], ['LAC'], ['MAQ'], ['MIP'], ['MOL'], ['MUE'], ['PAN'], ['PES'], ['PPR'], ['QU2'], ['REF'], ['RES'], ['SAL'], ['SEG'], ['TEL'], ['TRA'], ['AGU', 'BNA'], ['AGU', 'CHO'], ['AGU', 'HIL'], ['AGU', 'MET'], ['AGU', 'SIL'], ['AGU', 'VES'], ['AZU', 'BNA'], ['AZU', 'CHO'], ['AZU', 'DOM'], ['AZU', 'HIL'], ['AZU', 'MAD'], ['AZU', 'MET'], ['AZU', 'SIL'], ['AZU', 'VES'], ['BNA', 'CAN'], ['BNA', 'CAU'], ['BNA', 'CER'], ['BNA', 'CHO'], ['BNA', 'CUE'], ['BNA', 'DOM'], ['BNA', 'FID'], ['BNA', 'HIL'], ['BNA', 'HOT'], ['BNA', 'MAD'], ['BNA', 'MAN'], ['BNA', 'MET'], ['BNA', 'PAP'], ['BNA', 'PLS'], ['BNA', 'POS'], ['BNA', 'REP'], ['BNA', 'SIL'], ['BNA', 'TAB'], ['BNA', 'VES'], ['CAN', 'CHO'], ['CAN', 'MET'], ['CAN', 'SIL'], ['CAN', 'VES'], ['CAU', 'CHO'], ['CAU', 'VES'], ['CER', 'CHO'], ['CER', 'DOM'], ['CER', 'HIL'], ['CER', 'MAD'], ['CER', 'MET'], ['CER', 'REP'], ['CER', 'SIL'], ['CER', 'VES'], ['CHO', 'CUE'], ['CHO', 'DOM'], ['CHO', 'FID'], ['CHO', 'HIL'], ['CHO', 'HOT'], ['CHO', 'MAD'], ['CHO', 'MAN'], ['CHO', 'MET'], ['CHO', 'PAP'], ['CHO', 'PLS'], ['CHO', 'POS'], ['CHO', 'REP'], ['CHO', 'SIL'], ['CHO', 'VES'], ['CUE', 'HIL'], ['CUE', 'MAD'], ['CUE', 'MET'], ['CUE', 'SIL'], ['CUE', 'VES'], ['DOM', 'HIL'], ['DOM', 'HOT'], ['DOM', 'MAD'], ['DOM', 'MET'], ['DOM', 'PAP'], ['DOM', 'REP'], ['DOM', 'SIL'], ['DOM', 'VES'], ['FID', 'VES'], ['HIL', 'HOT'], ['HIL', 'MAD'], ['HIL', 'MET'], ['HIL', 'PAP'], ['HIL', 'PLS'], ['HIL', 'REP'], ['HIL', 'SIL'], ['HIL', 'VES'], ['HOT', 'MAD'], ['HOT', 'MET'], ['HOT', 'SIL'], ['HOT', 'VES'], ['MAD', 'MET'], ['MAD', 'PAP'], ['MAD', 'REP'], ['MAD', 'SIL'], ['MAD', 'VES'], ['MAN', 'VES'], ['MET', 'PAP'], ['MET', 'PLS'], ['MET', 'REP'], ['MET', 'SIL'], ['MET', 'VES'], ['PAP', 'REP'], ['PAP', 'SIL'], ['PAP', 'VES'], ['PLS', 'SIL'], ['PLS', 'VES'], ['POS', 'SIL'], ['POS', 'VES'], ['REP', 'SIL'], ['REP', 'VES'], ['SIL', 'VES'], ['TAB', 'VES']]


class AdHocExecutor(SimulationExecutor):
    """Executes ad-hoc disruption analysis across multiple sectors or subregions."""

    def __init__(self, model, parameters, disruption_type="sectors", subregion=None, results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
        self.disruption_type = disruption_type
        self.subregion = subregion

    def execute(self) -> List["Simulation"]:
        """Execute ad-hoc analysis and return list of simulations."""
        from disruptsc.model.caching_functions import load_cached_model

        # Save model state
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)

        # Get disrupted combinations and present values based on type
        if self.disruption_type == "sectors":
            disrupted_list = _get_disrupted_sector_list()
            present_values = list(set(self.model.firms.get_properties('region_sector', 'list')))
            logging.info(f"{len(disrupted_list)} sector combinations to test")
        else:  # subregions
            disrupted_list = _get_disrupted_subregion_list(self.subregion)
            present_subregions = self.model.firms.get_properties('subregions', 'list')
            present_values = list(set([s[self.subregion] for s in present_subregions]))
            logging.info(f"{len(disrupted_list)} {self.subregion} combinations to test")
        
        periods = [30, 90, 180]
        results = []
        
        for disrupted_targets in disrupted_list:
            if disrupted_targets == 'all':
                disrupted_targets = present_values

            # Add prefix for sectors and check against prefixed present values
            if self.disruption_type == "sectors":
                disrupted_targets = ['ECU_' + sector for sector in disrupted_targets]
                # For sectors, present_values already has ECU_ prefix
                check_values = present_values
            else:
                # For subregions, no prefix needed
                check_values = present_values

            # Skip if targets not present
            if any([target not in check_values for target in disrupted_targets]):
                continue

            # Log disruption type
            if self.disruption_type == "sectors":
                logging.info(f"")
                logging.info(f"=============== Disrupting sector #{disrupted_targets} ===============")
            else:
                logging.info(f"")
                logging.info(f"=============== Disrupting {self.subregion} #{disrupted_targets} ===============")

            # Load fresh model state
            model = load_cached_model(suffix)
            
            # Set disruption parameters based on type
            if self.disruption_type == "sectors":
                model.parameters.events[0]['region_sectors'] = disrupted_targets
            else:
                model.parameters.disruptions[0]['filter']['subregion_'+self.subregion] = disrupted_targets

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
                self.results_writer.write_ad_hoc_results(disrupted_targets, household_loss, country_loss,
                                                         household_loss_per_periods)

            # Clear simulation from memory immediately after processing
            del simulation
            del model
            gc.collect()
            
            # Don't accumulate simulation objects to prevent memory leaks
            # results.append(simulation)

        return results


def _get_disrupted_subregion_list(which_subregion: str) -> List:
    """Get the list of sector combinations to test."""
    # This would ideally be loaded from a config file
    # For now, keeping the original hardcoded list
    if which_subregion == "province":
        return [['AZUAY'], ['CAÑAR'], ['CHIMBORAZO'], ['COTOPAXI'], ['EL ORO'], ['ESMERALDAS'], ['GUAYAS'], ['LOJA'], ['LOS RIOS'], ['MANABI'], ['PICHINCHA'], ['TUNGURAHUA'], ['CARCHI', 'IMBABURA'], ['IMBABURA', 'NAPO'], ['IMBABURA', 'SUCUMBIOS'], ['ORELLANA', 'SUCUMBIOS']]
    if which_subregion == "canton":
        return [['AZUAY - CUENCA'], ['CAÑAR - AZOGUES'], ['COTOPAXI - LATACUNGA'], ['EL ORO - MACHALA'], ['ESMERALDAS - RIO VERDE'], ['GUAYAS - DURAN'], ['GUAYAS - GUAYAQUIL'], ['GUAYAS - SAMBORONDON'], ['LOJA - LOJA'], ['LOS RIOS - QUEVEDO'], ['MANABI - JARAMIJO'], ['MANABI - MANTA'], ['MANABI - PORTOVIEJO'], ['PICHINCHA - QUITO'], ['PICHINCHA - SANTO DOMINGO'], ['TUNGURAHUA - AMBATO'], ['BOLIVAR - GUARANDA', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - PENIPE'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - QUERO'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - SAN PEDRO DE PELILEO'], ['EL ORO - PASAJE', 'EL ORO - SANTA ROSA'], ['ESMERALDAS - ESMERALDAS', 'ESMERALDAS - QUININDE'], ['GUAYAS - EL TRIUNFO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['GUAYAS - MILAGRO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['IMBABURA - IBARRA', 'PICHINCHA - CAYAMBE'], ['PICHINCHA - MEJIA', 'PICHINCHA - RUMIÑAHUI'], ['GUAYAS - DAULE', 'GUAYAS - PEDRO CARBO', 'GUAYAS - SANTA LUCIA'], ['GUAYAS - DAULE', 'GUAYAS - SALITRE', 'LOS RIOS - BABAHOYO'], ['ESMERALDAS - LA CONCORDIA', 'MANABI - CHONE', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - SUCRE'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - JUNIN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - PICHINCHA', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - JIPIJAPA', 'MANABI - MONTECRISTI', 'MANABI - PICHINCHA', 'MANABI - SANTA ANA']]



