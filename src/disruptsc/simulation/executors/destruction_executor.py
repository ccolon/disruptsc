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


class DestructionExecutor(SimulationExecutor):
    """Executes ad-hoc disruption analysis across multiple sectors or subregions."""

    def __init__(self, model, parameters, target_types="sectors", subregion=None, results_writer=None):
        super().__init__(model, parameters)
        self.results_writer = results_writer
        self.target_types = target_types
        self.subregion = subregion

    def execute(self) -> List["Simulation"]:
        """Execute ad-hoc analysis and return list of simulations."""
        from disruptsc.model.utils.caching import load_cached_model

        # Save model state
        suffix = round(datetime.now().timestamp() * 1000)
        self.model.save_pickle(suffix)

        # Get disrupted combinations and present values based on type
        if self.target_types == "sector":
            disrupted_targets_list = _get_disrupted_sector_list()
            targets_in_model = self.model.firms.get_properties('sector', 'set')
            logging.info(f"{len(disrupted_targets_list)} sector combinations to test")
        elif self.target_types in ["canton", "province"]:
            disrupted_targets_list = _get_disrupted_subregion_list(self.subregion)
            targets_in_model = self.model.firms.get_subregions(self.subregion, 'set')
            logging.info(f"{len(disrupted_targets_list)} {self.subregion} combinations to test")
        elif self.target_types in ["canton_sector", "province_sector"]:
            disrupted_targets_list = _get_disrupted_subregion_sector_list(self.subregion)
            targets_in_model = self.model.firms.get_subregion_sectors(self.subregion, 'list')
            logging.info(f"{len(disrupted_targets_list)} {self.subregion}-sector combinations to test")

        periods = [30, 90, 180]
        results = []
        
        for disrupted_targets in disrupted_targets_list:
            # Handle existing logic for sectors and subregions
            if disrupted_targets == 'all':
                disrupted_targets = targets_in_model

            # Skip if targets not present
            if all([target not in targets_in_model for target in disrupted_targets]):
                logging.info(f"No targets is present in model: {disrupted_targets}")
                continue

            disrupted_targets_not_in_model = set(disrupted_targets) - set(targets_in_model)
            if len(disrupted_targets_not_in_model) > 0:
                logging.info(f"Skipping {disrupted_targets_not_in_model} - not present in model")
                disrupted_targets = list(set(disrupted_targets) & set(targets_in_model))

            # Log disruption type
            logging.info(f"")
            logging.info(f"=============== Disrupting {self.target_types} #{disrupted_targets} ===============")

            # Load fresh model state
            model = load_cached_model(suffix)
            
            # Set disruption parameters based on type
            model.parameters.disruptions[0]['filter'] = {}
            model.parameters.disruptions[0]['filter'][self.target_types] = disrupted_targets
            logging.info(model.parameters.disruptions[0])

            # Run simulation
            simulation = model.run_disruption(t_final=periods[-1])

            # Calculate losses
            household_loss = simulation.calculate_household_loss(model.household_table)
            household_loss_per_periods = simulation.calculate_household_loss(
                model.household_table, periods=periods
            )
            country_loss = simulation.calculate_country_loss()

            logging.info(f"Simulation terminated. "
                         f"Household loss: {household_loss_per_periods}. "
                         f"Country loss: {int(country_loss)} {self.parameters.monetary_units_in_model}.")

            # Write results if writer provided
            if self.results_writer:
                # Format results identifier based on disruption type
                results_identifier = flatten_to_str(disrupted_targets)
                    
                self.results_writer.write_destruction_results(results_identifier, household_loss, country_loss,
                                                              household_loss_per_periods)

            # Clear simulation from memory immediately after processing
            del simulation
            del model
            gc.collect()
            
            # Don't accumulate simulation objects to prevent memory leaks
            # results.append(simulation)

        return results


def flatten_to_str(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, (list, tuple)):
        return '_'.join(flatten_to_str(i) for i in x)
    else:
        return str(x)


def _get_disrupted_subregion_list(which_subregion: str) -> List:
    """Get the list of sector combinations to test."""
    # For now, keeping the original hardcoded list
    if which_subregion == "province":
        return [['AZUAY'], ['CAÑAR'], ['CHIMBORAZO'], ['COTOPAXI'], ['EL ORO'], ['ESMERALDAS'], ['GUAYAS'], ['LOJA'], ['LOS RIOS'], ['MANABI'], ['PICHINCHA'], ['TUNGURAHUA'], ['CARCHI', 'IMBABURA'], ['IMBABURA', 'NAPO'], ['IMBABURA', 'SUCUMBIOS'], ['ORELLANA', 'SUCUMBIOS']]
    if which_subregion == "canton":
        return [['AZUAY - CUENCA'], ['CAÑAR - AZOGUES'], ['COTOPAXI - LATACUNGA'], ['EL ORO - MACHALA'], ['ESMERALDAS - RIO VERDE'], ['GUAYAS - DURAN'], ['GUAYAS - GUAYAQUIL'], ['GUAYAS - SAMBORONDON'], ['LOJA - LOJA'], ['LOS RIOS - QUEVEDO'], ['MANABI - JARAMIJO'], ['MANABI - MANTA'], ['MANABI - PORTOVIEJO'], ['PICHINCHA - QUITO'], ['PICHINCHA - SANTO DOMINGO'], ['TUNGURAHUA - AMBATO'], ['BOLIVAR - GUARANDA', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - PENIPE'], ['CHIMBORAZO - GUANO', 'CHIMBORAZO - RIOBAMBA'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - QUERO'], ['CHIMBORAZO - GUANO', 'TUNGURAHUA - SAN PEDRO DE PELILEO'], ['EL ORO - PASAJE', 'EL ORO - SANTA ROSA'], ['ESMERALDAS - ESMERALDAS', 'ESMERALDAS - QUININDE'], ['GUAYAS - EL TRIUNFO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['GUAYAS - MILAGRO', 'GUAYAS - SAN JACINTO DE YAGUACHI'], ['IMBABURA - IBARRA', 'PICHINCHA - CAYAMBE'], ['PICHINCHA - MEJIA', 'PICHINCHA - RUMIÑAHUI'], ['GUAYAS - DAULE', 'GUAYAS - PEDRO CARBO', 'GUAYAS - SANTA LUCIA'], ['GUAYAS - DAULE', 'GUAYAS - SALITRE', 'LOS RIOS - BABAHOYO'], ['ESMERALDAS - LA CONCORDIA', 'MANABI - CHONE', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - SUCRE'], ['MANABI - CHONE', 'MANABI - EL CARMEN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - JUNIN', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - CHONE', 'MANABI - PICHINCHA', 'MANABI - ROCAFUERTE', 'MANABI - TOSAGUA'], ['MANABI - JIPIJAPA', 'MANABI - MONTECRISTI', 'MANABI - PICHINCHA', 'MANABI - SANTA ANA']]


def _get_disrupted_subregion_sector_list(which_subregion: str) -> List:
    """Get strategic list of (subregion, sector) combinations to test."""
    if which_subregion == "province":
        return [[('AZUAY', 'ELE')], [('AZUAY', 'ADP'), ('AZUAY', 'CAR')], [('AZUAY', 'AYG'), ('AZUAY', 'COM')], [('AZUAY', 'CON'), ('AZUAY', 'EDU')], [('AZUAY', 'FIN'), ('AZUAY', 'INM'), ('AZUAY', 'SAL')], [('CAÑAR', 'PPR')], [('CHIMBORAZO', 'CON'), ('CHIMBORAZO', 'MIP')], [('COTOPAXI', 'CON'), ('COTOPAXI', 'EDU'), ('COTOPAXI', 'TRA')], [('EL ORO', 'CON'), ('EL ORO', 'FRV')], [('EL ORO', 'ADP'), ('EL ORO', 'ALD'), ('EL ORO', 'COM'), ('EL ORO', 'FRT')], [('ESMERALDAS', 'MIP')], [('GUAYAS', 'ADP')], [('GUAYAS', 'COM')], [('GUAYAS', 'CON')], [('GUAYAS', 'EDU')], [('GUAYAS', 'INM')], [('GUAYAS', 'PES')], [('GUAYAS', 'REF')], [('GUAYAS', 'RES')], [('GUAYAS', 'SAL')], [('GUAYAS', 'TEL')], [('GUAYAS', 'TRA')], [('GUAYAS', 'ADM'), ('GUAYAS', 'ASO')], [('GUAYAS', 'AGU'), ('GUAYAS', 'BAL')], [('GUAYAS', 'ALD'), ('GUAYAS', 'ELE')], [('GUAYAS', 'BNA'), ('GUAYAS', 'FIN')], [('GUAYAS', 'CAN'), ('GUAYAS', 'MOL')], [('GUAYAS', 'CAR'), ('GUAYAS', 'FRT')], [('GUAYAS', 'CHO'), ('GUAYAS', 'LAC')], [('GUAYAS', 'FRV'), ('GUAYAS', 'MAQ')], [('GUAYAS', 'MET'), ('GUAYAS', 'PPR')], [('GUAYAS', 'PAN'), ('GUAYAS', 'SEG')], [('GUAYAS', 'CIN'), ('GUAYAS', 'DEM'), ('GUAYAS', 'MIP')], [('GUAYAS', 'GAN'), ('GUAYAS', 'PAP'), ('GUAYAS', 'QU2')], [('LOJA', 'CON'), ('LOJA', 'MIP')], [('LOS RIOS', 'FRT')], [('LOS RIOS', 'ASO'), ('LOS RIOS', 'CIN'), ('LOS RIOS', 'CON')], [('MANABI', 'CON')], [('MANABI', 'PPR')], [('MANABI', 'AYG'), ('MANABI', 'FRV')], [('MANABI', 'EDU'), ('MANABI', 'TRA')], [('PICHINCHA', 'ADP')], [('PICHINCHA', 'CAR')], [('PICHINCHA', 'COM')], [('PICHINCHA', 'CON')], [('PICHINCHA', 'EDU')], [('PICHINCHA', 'GAN')], [('PICHINCHA', 'INM')], [('PICHINCHA', 'MIP')], [('PICHINCHA', 'RES')], [('PICHINCHA', 'SAL')], [('PICHINCHA', 'TEL')], [('PICHINCHA', 'TRA')], [('PICHINCHA', 'ADM'), ('PICHINCHA', 'ALD')], [('PICHINCHA', 'ASO'), ('PICHINCHA', 'BNA')], [('PICHINCHA', 'AYG'), ('PICHINCHA', 'FIN')], [('PICHINCHA', 'CHO'), ('PICHINCHA', 'MAQ')], [('PICHINCHA', 'BAL'), ('PICHINCHA', 'CIN'), ('PICHINCHA', 'PAN')], [('PICHINCHA', 'DEM'), ('PICHINCHA', 'ELE'), ('PICHINCHA', 'QU2')], [('PICHINCHA', 'HIL'), ('PICHINCHA', 'LAC'), ('PICHINCHA', 'SEG')], [('PICHINCHA', 'DOM'), ('PICHINCHA', 'MAD'), ('PICHINCHA', 'MUE'), ('PICHINCHA', 'REF')], [('TUNGURAHUA', 'ADP'), ('TUNGURAHUA', 'CON'), ('TUNGURAHUA', 'FRT')]]
    
    elif which_subregion == "canton":
        return [[('AZUAY - CUENCA', 'ELE')], [('AZUAY - CUENCA', 'ADP'), ('AZUAY - CUENCA', 'CAR')], [('AZUAY - CUENCA', 'AYG'), ('AZUAY - CUENCA', 'COM')], [('AZUAY - CUENCA', 'CON'), ('AZUAY - CUENCA', 'EDU')], [('AZUAY - CUENCA', 'FIN'), ('AZUAY - CUENCA', 'INM'), ('AZUAY - CUENCA', 'SAL')], [('CAÑAR - AZOGUES', 'PPR')], [('GUAYAS - GUAYAQUIL', 'ADP')], [('GUAYAS - GUAYAQUIL', 'COM')], [('GUAYAS - GUAYAQUIL', 'CON')], [('GUAYAS - GUAYAQUIL', 'EDU')], [('GUAYAS - GUAYAQUIL', 'INM')], [('GUAYAS - GUAYAQUIL', 'PES')], [('GUAYAS - GUAYAQUIL', 'REF')], [('GUAYAS - GUAYAQUIL', 'SAL')], [('GUAYAS - GUAYAQUIL', 'TEL')], [('GUAYAS - GUAYAQUIL', 'TRA')], [('GUAYAS - GUAYAQUIL', 'ADM'), ('GUAYAS - GUAYAQUIL', 'ASO')], [('GUAYAS - GUAYAQUIL', 'AGU'), ('GUAYAS - GUAYAQUIL', 'ELE')], [('GUAYAS - GUAYAQUIL', 'BAL'), ('GUAYAS - GUAYAQUIL', 'BNA')], [('GUAYAS - GUAYAQUIL', 'CAR'), ('GUAYAS - GUAYAQUIL', 'FIN')], [('GUAYAS - GUAYAQUIL', 'CHO'), ('GUAYAS - GUAYAQUIL', 'LAC')], [('GUAYAS - GUAYAQUIL', 'FRT'), ('GUAYAS - GUAYAQUIL', 'MAQ')], [('GUAYAS - GUAYAQUIL', 'HIL'), ('GUAYAS - GUAYAQUIL', 'RES')], [('GUAYAS - GUAYAQUIL', 'MOL'), ('GUAYAS - GUAYAQUIL', 'PAN')], [('GUAYAS - GUAYAQUIL', 'ALD'), ('GUAYAS - GUAYAQUIL', 'MIP'), ('GUAYAS - GUAYAQUIL', 'SEG')], [('LOJA - LOJA', 'CON'), ('LOJA - LOJA', 'MIP')], [('AZUAY - CUENCA', 'MUE'), ('GUAYAS - NARANJAL', 'FRV'), ('GUAYAS - SAN JACINTO DE YAGUACHI', 'PPR'), ('GUAYAS - MILAGRO', 'MET')], [('GUAYAS - SAN JACINTO DE YAGUACHI', 'CON'), ('GUAYAS - SAMBORONDON', 'EDU'), ('GUAYAS - SAMBORONDON', 'INM')], [('GUAYAS - GUAYAQUIL', 'AYG'), ('GUAYAS - GUAYAQUIL', 'DEM'), ('GUAYAS - GUAYAQUIL', 'QU2'), ('GUAYAS - DAULE', 'CON')], [('LOS RIOS - QUEVEDO', 'FRT')], [('BOLIVAR - CALUMA', 'MAQ'), ('BOLIVAR - GUARANDA', 'ADM'), ('TUNGURAHUA - AMBATO', 'FRT')], [('CHIMBORAZO - GUANO', 'CON'), ('CHIMBORAZO - GUANO', 'MIP'), ('TUNGURAHUA - SAN PEDRO DE PELILEO', 'CON')], [('TUNGURAHUA - AMBATO', 'COM'), ('TUNGURAHUA - AMBATO', 'EDU'), ('COTOPAXI - SALCEDO', 'CON'), ('COTOPAXI - LATACUNGA', 'TRA')], [('PICHINCHA - QUITO', 'ADP')], [('PICHINCHA - QUITO', 'CAR')], [('PICHINCHA - QUITO', 'COM')], [('PICHINCHA - QUITO', 'CON')], [('PICHINCHA - QUITO', 'EDU')], [('PICHINCHA - QUITO', 'GAN')], [('PICHINCHA - QUITO', 'INM')], [('PICHINCHA - QUITO', 'MIP')], [('PICHINCHA - QUITO', 'SAL')], [('PICHINCHA - QUITO', 'TEL')], [('PICHINCHA - QUITO', 'TRA')], [('PICHINCHA - QUITO', 'ADM'), ('PICHINCHA - QUITO', 'ALD')], [('PICHINCHA - QUITO', 'ASO'), ('PICHINCHA - QUITO', 'BNA')], [('PICHINCHA - QUITO', 'AYG'), ('PICHINCHA - QUITO', 'FIN')], [('PICHINCHA - QUITO', 'BAL'), ('PICHINCHA - QUITO', 'RES')], [('PICHINCHA - QUITO', 'ELE'), ('PICHINCHA - QUITO', 'MAQ')], [('PICHINCHA - QUITO', 'CHO'), ('PICHINCHA - QUITO', 'LAC'), ('PICHINCHA - QUITO', 'PAN')], [('PICHINCHA - QUITO', 'DEM'), ('PICHINCHA - QUITO', 'QU2'), ('PICHINCHA - QUITO', 'SEG')], [('PICHINCHA - QUITO', 'HIL'), ('PICHINCHA - QUITO', 'MAD'), ('PICHINCHA - QUITO', 'SIL'), ('PICHINCHA - PEDRO VICENTE MALDONADO', 'GAN')], [('PICHINCHA - QUITO', 'MET'), ('PICHINCHA - QUITO', 'MUE'), ('PICHINCHA - QUITO', 'REF'), ('PICHINCHA - CAYAMBE', 'COM')], [('EL ORO - MACHALA', 'ADP'), ('EL ORO - MACHALA', 'COM'), ('EL ORO - MACHALA', 'CON'), ('EL ORO - MACHALA', 'INM')], [('MANABI - PORTOVIEJO', 'CON'), ('MANABI - PORTOVIEJO', 'EDU'), ('MANABI - PORTOVIEJO', 'PPR'), ('MANABI - PORTOVIEJO', 'SAL')], [('ESMERALDAS - RIO VERDE', 'MIP')], [('MANABI - JARAMIJO', 'PPR')], [('MANABI - JIPIJAPA', 'CON'), ('MANABI - JIPIJAPA', 'EDU'), ('MANABI - MONTECRISTI', 'AYG'), ('MANABI - MANTA', 'CON')]]
    
    return []



