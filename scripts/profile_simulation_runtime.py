#!/usr/bin/env python3
"""
Simulation Runtime Profiling Script for DisruptSC

This script provides detailed performance analysis of simulation time steps,
identifying bottlenecks in agent behavior, transport, and network operations.
"""

import cProfile
import time
import logging
import pstats
from pathlib import Path
import sys
from typing import Dict, Any
import io
import argparse
from collections import defaultdict

# Add src to Python path
src_path = Path(__file__).parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from disruptsc import paths
from disruptsc.model.caching_functions import generate_cache_parameters_from_command_line_argument
from disruptsc.parameters import Parameters
from disruptsc.model.model import Model
from disruptsc.model.profiling_utils import ModelProfiler, ProfiledModel


class SimulationRuntimeProfiler:
    """Class to profile simulation runtime with detailed time step analysis."""
    
    def __init__(self):
        self.time_step_timings = defaultdict(dict)
        self.phase_timings = defaultdict(list)
        self.agent_timings = defaultdict(list)
        
    def profile_time_step(self, model, time_step, simulation, enable_cprofile=False):
        """Profile a single time step with detailed phase breakdown."""
        
        if enable_cprofile:
            profiler = cProfile.Profile()
            profiler.enable()
        
        step_start = time.perf_counter()
        
        # Phase 1: Disruption Application
        phase_start = time.perf_counter()
        available_transport_network = model.transport_network
        if model.disruption_list:
            available_transport_network = model.apply_disruption(time_step)
        self.time_step_timings[time_step]['disruption_application'] = time.perf_counter() - phase_start
        
        # Phase 2: Order Retrieval & Reconstruction Market
        phase_start = time.perf_counter()
        model.firms.retrieve_orders(model.sc_network)
        if model.reconstruction_market:
            model.reconstruction_market.evaluate_demand_to_firm(model.firms)
            model.reconstruction_market.send_orders(model.firms)
        self.time_step_timings[time_step]['order_retrieval'] = time.perf_counter() - phase_start
        
        # Phase 3: Production Planning
        phase_start = time.perf_counter()
        model.firms.plan_production(model.sc_network, model.parameters.propagate_input_price_change)
        self.time_step_timings[time_step]['production_planning'] = time.perf_counter() - phase_start
        
        # Phase 4: Purchase Planning
        phase_start = time.perf_counter()
        model.firms.plan_purchase(model.parameters.adaptive_inventories, model.parameters.adaptive_supplier_weight)
        self.time_step_timings[time_step]['purchase_planning'] = time.perf_counter() - phase_start
        
        # Phase 5: Purchase Orders
        phase_start = time.perf_counter()
        model.households.send_purchase_orders(model.sc_network)
        model.countries.send_purchase_orders(model.sc_network)
        model.firms.send_purchase_orders(model.sc_network)
        self.time_step_timings[time_step]['purchase_orders'] = time.perf_counter() - phase_start
        
        # Phase 6: Production
        phase_start = time.perf_counter()
        model.firms.produce()
        self.time_step_timings[time_step]['production'] = time.perf_counter() - phase_start
        
        # Phase 7: Delivery (Countries)
        phase_start = time.perf_counter()
        model.countries.deliver(model.sc_network, model.transport_network, available_transport_network,
                               model.parameters.sectors_no_transport_network,
                               model.parameters.rationing_mode, model.parameters.with_transport,
                               model.parameters.transport_to_households, model.parameters.capacity_constraint,
                               model.parameters.monetary_units_in_model, model.parameters.cost_repercussion_mode,
                               model.parameters.price_increase_threshold, model.parameters.transport_cost_noise_level,
                               model.parameters.use_route_cache)
        self.time_step_timings[time_step]['countries_delivery'] = time.perf_counter() - phase_start
        
        # Phase 8: Delivery (Firms)
        phase_start = time.perf_counter()
        model.firms.deliver(model.sc_network, model.transport_network, available_transport_network,
                           model.parameters.sectors_no_transport_network,
                           model.parameters.rationing_mode, model.parameters.with_transport,
                           model.parameters.transport_to_households, model.parameters.capacity_constraint,
                           model.parameters.monetary_units_in_model, model.parameters.cost_repercussion_mode,
                           model.parameters.price_increase_threshold, model.parameters.transport_cost_noise_level,
                           model.parameters.use_route_cache)
        self.time_step_timings[time_step]['firms_delivery'] = time.perf_counter() - phase_start
        
        # Phase 9: Reconstruction Market Distribution
        phase_start = time.perf_counter()
        if model.reconstruction_market:
            model.reconstruction_market.distribute_new_capital(model.firms)
        self.time_step_timings[time_step]['reconstruction_distribution'] = time.perf_counter() - phase_start
        
        # Phase 10: Transport Flow Computing
        phase_start = time.perf_counter()
        if (simulation.type not in ['criticality']) and (time_step in [0, 1]):
            simulation.transport_network_data += model.transport_network.compute_flow_per_segment(time_step)
        self.time_step_timings[time_step]['transport_flow_computing'] = time.perf_counter() - phase_start
        
        # Phase 11: Product Reception
        phase_start = time.perf_counter()
        model.households.receive_products(model.sc_network, model.transport_network,
                                         model.parameters.sectors_no_transport_network,
                                         model.parameters.transport_to_households)
        model.countries.receive_products(model.sc_network, model.transport_network,
                                        model.parameters.sectors_no_transport_network)
        model.firms.receive_products(model.sc_network, model.transport_network,
                                    model.parameters.sectors_no_transport_network)
        self.time_step_timings[time_step]['product_reception'] = time.perf_counter() - phase_start
        
        # Phase 12: Transport Cleanup & Profit Evaluation
        phase_start = time.perf_counter()
        model.transport_network.check_no_uncollected_shipment()
        model.transport_network.reset_loads()
        model.firms.evaluate_profit(model.sc_network)
        self.time_step_timings[time_step]['transport_cleanup_profit'] = time.perf_counter() - phase_start
        
        # Phase 13: State Updates
        phase_start = time.perf_counter()
        model.transport_network.update_road_disruption_state()
        model.firms.update_disrupted_production_capacity()
        self.time_step_timings[time_step]['state_updates'] = time.perf_counter() - phase_start
        
        # Phase 14: Data Storage
        phase_start = time.perf_counter()
        model.store_agent_data(time_step, simulation)
        model.store_sc_network_data(time_step, simulation)
        self.time_step_timings[time_step]['data_storage'] = time.perf_counter() - phase_start
        
        # Total time step duration
        total_duration = time.perf_counter() - step_start
        self.time_step_timings[time_step]['total'] = total_duration
        
        if enable_cprofile:
            profiler.disable()
            return profiler, total_duration
        
        return None, total_duration
    
    def print_time_step_summary(self, time_step=None):
        """Print detailed summary for a specific time step or all time steps."""
        print("\n" + "="*80)
        print("🕐 SIMULATION TIME STEP PROFILING SUMMARY")
        print("="*80)
        
        if time_step is not None:
            # Print specific time step
            if time_step in self.time_step_timings:
                self._print_single_time_step(time_step)
            else:
                print(f"No data available for time step {time_step}")
        else:
            # Print all time steps
            for ts in sorted(self.time_step_timings.keys()):
                self._print_single_time_step(ts)
                print("-" * 80)
    
    def _print_single_time_step(self, time_step):
        """Print profiling data for a single time step."""
        data = self.time_step_timings[time_step]
        total_time = data.get('total', 0)
        
        print(f"\n📊 TIME STEP {time_step} (Total: {total_time:.3f}s)")
        print(f"{'Phase':<35} {'Duration(s)':<12} {'% of Step':<10}")
        print("-" * 60)
        
        # Sort phases by duration
        phases = [(k, v) for k, v in data.items() if k != 'total']
        phases.sort(key=lambda x: x[1], reverse=True)
        
        for phase, duration in phases:
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            print(f"{phase.replace('_', ' ').title():<35} {duration:<12.3f} {percentage:<10.1f}%")
    
    def get_phase_statistics(self):
        """Get aggregated statistics across all time steps."""
        phase_stats = defaultdict(lambda: {'durations': [], 'total': 0, 'count': 0})
        
        for time_step, phases in self.time_step_timings.items():
            for phase, duration in phases.items():
                if phase != 'total':
                    phase_stats[phase]['durations'].append(duration)
                    phase_stats[phase]['total'] += duration
                    phase_stats[phase]['count'] += 1
        
        # Calculate statistics
        stats = {}
        for phase, data in phase_stats.items():
            durations = data['durations']
            if durations:
                stats[phase] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        
        return stats
    
    def print_phase_statistics(self):
        """Print aggregated phase statistics."""
        stats = self.get_phase_statistics()
        
        print("\n" + "="*80)
        print("📈 AGGREGATED PHASE STATISTICS")
        print("="*80)
        print(f"{'Phase':<35} {'Count':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10}")
        print("-" * 80)
        
        # Sort by total time
        sorted_phases = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for phase, data in sorted_phases:
            print(f"{phase.replace('_', ' ').title():<35} {data['count']:<8} {data['total']:<10.3f} "
                  f"{data['average']:<10.3f} {data['min']:<10.3f} {data['max']:<10.3f}")
        
        print("="*80)
        total_time = sum(data['total'] for data in stats.values())
        print(f"Total simulation time: {total_time:.3f}s")
        print("="*80)


def profile_simulation_runtime(scope: str = "Testkistan", 
                              simulation_type: str = "initial_state",
                              time_steps: int = 5,
                              cache_type: str = None,
                              enable_cprofile: bool = False,
                              enable_model_profiler: bool = True) -> Dict[str, Any]:
    """
    Profile simulation runtime with detailed time step analysis.
    
    Parameters
    ----------
    scope : str
        Region scope to use for profiling
    simulation_type : str
        Type of simulation to run
    time_steps : int
        Number of time steps to profile
    cache_type : str
        Cache configuration to use
    enable_cprofile : bool
        Whether to enable cProfile for detailed function analysis
    enable_model_profiler : bool
        Whether to enable the model's built-in profiler
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing profiling results
    """
    
    print(f"\n🚀 Starting simulation runtime profiling")
    print(f"Scope: {scope}, Type: {simulation_type}, Steps: {time_steps}")
    
    # Initialize profiler
    runtime_profiler = SimulationRuntimeProfiler()
    
    try:
        # Setup model (reuse from setup profiling)
        print("\n⚙️  Setting up model...")
        setup_start = time.perf_counter()
        
        cache_parameters = generate_cache_parameters_from_command_line_argument(cache_type)
        parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)
        parameters.initialize_exports()
        parameters.adjust_logging_behavior()
        
        model = Model(parameters)
        model.setup_transport_network(cache_parameters['transport_network'], parameters.with_transport)
        model.setup_agents(cache_parameters['agents'])
        model.setup_sc_network(cache_parameters['sc_network'])
        model.set_initial_conditions()
        model.setup_logistic_routes(cache_parameters['logistic_routes'], parameters.with_transport)
        
        setup_duration = time.perf_counter() - setup_start
        print(f"✅ Model setup completed in {setup_duration:.2f}s")
        
        # Enable model profiler if requested
        profiler_context = None
        if enable_model_profiler:
            profiler_context = ProfiledModel(clear_previous=True)
            profiler_context.__enter__()
        
        # Run simulation with profiling
        print(f"\n🏃 Running {simulation_type} simulation with {time_steps} time steps...")
        
        if simulation_type == "initial_state":
            # For initial state, we run a single step
            from disruptsc.simulation.simulation import Simulation
            simulation = Simulation("initial_state")
            
            # Profile the single time step
            cprofile_data, duration = runtime_profiler.profile_time_step(
                model, 0, simulation, enable_cprofile=enable_cprofile
            )
            
            print(f"✅ Initial state simulation completed in {duration:.3f}s")
            
        elif simulation_type == "disruption":
            # Run disruption simulation
            simulation_start = time.perf_counter()
            total_cprofile_data = []
            
            for step in range(time_steps):
                print(f"  📊 Profiling time step {step}...")
                
                if step == 0:
                    # Initialize simulation
                    from disruptsc.simulation.simulation import Simulation
                    simulation = Simulation("event")
                    
                    # Get disruptions
                    from disruptsc.disruption.disruption import DisruptionList
                    model.disruption_list = DisruptionList.from_disruptions_parameter(
                        parameters.disruptions, parameters.monetary_units_in_model,
                        model.transport_edges, model.firm_table, model.firms
                    )
                
                cprofile_data, duration = runtime_profiler.profile_time_step(
                    model, step, simulation, enable_cprofile=enable_cprofile
                )
                
                if cprofile_data:
                    total_cprofile_data.append((step, cprofile_data))
                
                print(f"    ⏱️  Time step {step}: {duration:.3f}s")
                
                # Check for early termination
                if hasattr(model, 'is_back_to_equilibrium') and model.is_back_to_equilibrium:
                    print(f"    🎯 Equilibrium reached at step {step}")
                    break
            
            total_simulation_time = time.perf_counter() - simulation_start
            print(f"✅ Disruption simulation completed in {total_simulation_time:.3f}s")
        
        # Print profiling results
        runtime_profiler.print_time_step_summary()
        runtime_profiler.print_phase_statistics()
        
        # Print model profiler results if enabled
        if enable_model_profiler and profiler_context:
            profiler_context.__exit__(None, None, None)
            print("\n🔍 MODEL PROFILER RESULTS")
            ModelProfiler.print_summary()
        
        # Print cProfile results if enabled
        if enable_cprofile and 'total_cprofile_data' in locals():
            print("\n🔬 DETAILED FUNCTION PROFILING")
            print("="*80)
            for step, cprofile_data in total_cprofile_data:
                print(f"\n⏱️  Time Step {step} - Top 10 Functions")
                print("-" * 60)
                s = io.StringIO()
                stats = pstats.Stats(cprofile_data, stream=s).sort_stats('cumulative')
                stats.print_stats(10)
                print(s.getvalue())
        
        return {
            "time_step_timings": dict(runtime_profiler.time_step_timings),
            "phase_statistics": runtime_profiler.get_phase_statistics(),
            "setup_duration": setup_duration,
            "model": model
        }
        
    except Exception as e:
        print(f"\n❌ Error during simulation profiling: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile DisruptSC simulation runtime")
    parser.add_argument("--scope", default="Testkistan", help="Region scope to profile")
    parser.add_argument("--simulation-type", default="initial_state", 
                       choices=["initial_state", "disruption"], 
                       help="Type of simulation to profile")
    parser.add_argument("--time-steps", type=int, default=5, 
                       help="Number of time steps to profile (for disruption simulation)")
    parser.add_argument("--cache", help="Cache configuration")
    parser.add_argument("--enable-cprofile", action="store_true", 
                       help="Enable detailed function profiling")
    parser.add_argument("--disable-model-profiler", action="store_true",
                       help="Disable model's built-in profiler")
    
    args = parser.parse_args()
    
    profile_simulation_runtime(
        scope=args.scope,
        simulation_type=args.simulation_type,
        time_steps=args.time_steps,
        cache_type=args.cache,
        enable_cprofile=args.enable_cprofile,
        enable_model_profiler=not args.disable_model_profiler
    )