#!/usr/bin/env python3
"""
Model Setup Profiling Script for DisruptSC

This script provides detailed timing analysis of the model setup phase,
identifying time-consuming steps and potential optimization opportunities.
"""

import cProfile
import time
import logging
import pstats
from pathlib import Path
import sys
from typing import Dict, Any
from functools import wraps
import io

# Add src to Python path
src_path = Path(__file__).parents[1] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from disruptsc import paths
from disruptsc.model.utils.caching import generate_cache_parameters_from_command_line_argument
from disruptsc.parameters import Parameters
from disruptsc.model.model import Model


class SetupProfiler:
    """Class to profile model setup with detailed timing information."""
    
    def __init__(self):
        self.timings = {}
        self.start_time = None
        
    def start_timer(self, phase_name: str):
        """Start timing a phase."""
        self.start_time = time.perf_counter()
        print(f"🔄 Starting: {phase_name}")
        
    def end_timer(self, phase_name: str):
        """End timing a phase and record the duration."""
        if self.start_time is None:
            return
        duration = time.perf_counter() - self.start_time
        self.timings[phase_name] = duration
        print(f"✅ Completed: {phase_name} ({duration:.2f}s)")
        self.start_time = None
        
    def print_summary(self):
        """Print a summary of all timing results."""
        print("\n" + "="*60)
        print("📊 MODEL SETUP PROFILING SUMMARY")
        print("="*60)
        
        total_time = sum(self.timings.values())
        
        # Sort by duration (descending)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for phase, duration in sorted_timings:
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{phase:<35} {duration:>8.2f}s ({percentage:>5.1f}%)")
            
        print("-" * 60)
        print(f"{'TOTAL SETUP TIME':<35} {total_time:>8.2f}s (100.0%)")
        print("="*60)


def profile_model_setup(scope: str = "Testkistan", cache_type: str = None, 
                       enable_cprofile: bool = True) -> Dict[str, Any]:
    """
    Profile the complete model setup process.
    
    Parameters
    ----------
    scope : str
        Region scope to use for profiling (default: Testkistan for speed)
    cache_type : str
        Cache configuration to use
    enable_cprofile : bool
        Whether to enable cProfile for detailed function-level profiling
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing timing results and profiling data
    """
    
    # Initialize profiler
    setup_profiler = SetupProfiler()
    
    # Optional cProfile for detailed function analysis
    if enable_cprofile:
        profiler = cProfile.Profile()
        profiler.enable()
    
    print(f"\n🚀 Starting model setup profiling for scope: {scope}")
    
    try:
        # Phase 1: Parameter Loading
        setup_profiler.start_timer("1. Parameter Loading")
        cache_parameters = generate_cache_parameters_from_command_line_argument(cache_type)
        parameters = Parameters.load_parameters(paths.PARAMETER_FOLDER, scope)
        parameters.initialize_exports()
        parameters.adjust_logging_behavior()
        setup_profiler.end_timer("1. Parameter Loading")
        
        # Phase 2: Model Initialization
        setup_profiler.start_timer("2. Model Initialization")
        model = Model(parameters)
        setup_profiler.end_timer("2. Model Initialization")
        
        # Phase 3: Transport Network Setup
        setup_profiler.start_timer("3. Transport Network Setup")
        model.setup_transport_network(cache_parameters['transport_network'], parameters.with_transport)
        setup_profiler.end_timer("3. Transport Network Setup")
        
        # Phase 4: Agent Setup (detailed sub-phases)
        setup_profiler.start_timer("4. Agent Setup (Total)")
        
        # Sub-phase 4a: MRIO and Sector Loading
        setup_profiler.start_timer("4a. MRIO & Sector Loading")
        if not cache_parameters['agents']:
            model._prepare_mrio_and_sectors()
            setup_profiler.end_timer("4a. MRIO & Sector Loading")
            
            # Sub-phase 4b: Sector Filtering
            setup_profiler.start_timer("4b. Sector Filtering")
            filtered_industries = model._filter_sectors()
            setup_profiler.end_timer("4b. Sector Filtering")
            
            # Sub-phase 4c: Firm Setup
            setup_profiler.start_timer("4c. Firm Setup")
            present_sectors, present_region_sectors, flow_types_to_export = model.setup_firms(filtered_industries)
            setup_profiler.end_timer("4c. Firm Setup")
            
            # Sub-phase 4d: Household Setup
            setup_profiler.start_timer("4d. Household Setup")
            model.setup_households(present_region_sectors)
            setup_profiler.end_timer("4d. Household Setup")
            
            # Sub-phase 4e: Country Setup
            setup_profiler.start_timer("4e. Country Setup")
            model.setup_countries()
            setup_profiler.end_timer("4e. Country Setup")
            
            # Sub-phase 4f: Agent Caching
            setup_profiler.start_timer("4f. Agent Caching")
            model._cache_agent_data(present_sectors, present_region_sectors, flow_types_to_export)
            setup_profiler.end_timer("4f. Agent Caching")
        else:
            model.setup_agents(cache_parameters['agents'])
            setup_profiler.end_timer("4a. MRIO & Sector Loading")
            
        # Locate agents on transport network
        setup_profiler.start_timer("4g. Agent Location on Network")
        model.transport_network.locate_firms_on_nodes(model.firms)
        model.transport_network.locate_households_on_nodes(model.households)
        model.agents_initialized = True
        setup_profiler.end_timer("4g. Agent Location on Network")
        
        setup_profiler.end_timer("4. Agent Setup (Total)")
        
        # Phase 5: Supply Chain Network Setup
        setup_profiler.start_timer("5. Supply Chain Network Setup")
        
        if not cache_parameters['sc_network']:
            # Sub-phase 5a: Network Creation
            setup_profiler.start_timer("5a. SC Network Creation")
            model.setup_sc_network(cache_parameters['sc_network'])
            setup_profiler.end_timer("5a. SC Network Creation")
        else:
            model.setup_sc_network(cache_parameters['sc_network'])
            
        setup_profiler.end_timer("5. Supply Chain Network Setup")
        
        # Phase 6: Initial Conditions
        setup_profiler.start_timer("6. Initial Conditions Setup")
        model.set_initial_conditions()
        setup_profiler.end_timer("6. Initial Conditions Setup")
        
        # Phase 7: Logistic Routes Setup
        setup_profiler.start_timer("7. Logistic Routes Setup")
        model.setup_logistic_routes(cache_parameters['logistic_routes'], parameters.with_transport)
        setup_profiler.end_timer("7. Logistic Routes Setup")
        
        print(f"\n✅ Model setup completed successfully!")
        
        # Print timing summary
        setup_profiler.print_summary()
        
        # Print model statistics
        print("\n📈 MODEL STATISTICS")
        print("="*60)
        print(f"Number of firms:        {len(model.firms):>10,}")
        print(f"Number of households:   {len(model.households):>10,}")
        print(f"Number of countries:    {len(model.countries):>10,}")
        print(f"Commercial links:       {model.sc_network.number_of_edges():>10,}")
        print(f"Transport nodes:        {len(model.transport_nodes):>10,}")
        print(f"Transport edges:        {len(model.transport_edges):>10,}")
        
        # Disable cProfile and generate report
        results = {"timings": setup_profiler.timings, "model": model}
        
        if enable_cprofile:
            profiler.disable()
            
            # Generate cProfile report
            print("\n🔍 DETAILED FUNCTION PROFILING (Top 20)")
            print("="*60)
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            print(s.getvalue())
            
            results["cprofile_stats"] = stats
            
        return results
        
    except Exception as e:
        print(f"\n❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "timings": setup_profiler.timings}


def profile_setup_methods():
    """Profile setup with different cache configurations."""
    print("🎯 PROFILING MODEL SETUP WITH DIFFERENT CACHE CONFIGURATIONS")
    print("="*80)
    
    configurations = [
        ("No Cache", None),
        ("Transport Cache", "same_transport_network_new_agents"),
        ("Agents Cache", "same_agents_new_sc_network"),
        ("SC Network Cache", "same_sc_network_new_logistic_routes"),
        ("Full Cache", "same_logistic_routes")
    ]
    
    results = {}
    
    for config_name, cache_type in configurations:
        print(f"\n{'='*20} {config_name} {'='*20}")
        result = profile_model_setup(scope="Testkistan", cache_type=cache_type, enable_cprofile=False)
        if "timings" in result:
            total_time = sum(result["timings"].values())
            results[config_name] = total_time
            print(f"Total setup time: {total_time:.2f}s")
    
    # Compare results
    print(f"\n📊 CACHE CONFIGURATION COMPARISON")
    print("="*60)
    for config_name, total_time in sorted(results.items(), key=lambda x: x[1]):
        print(f"{config_name:<25} {total_time:>8.2f}s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile DisruptSC model setup")
    parser.add_argument("--scope", default="Testkistan", help="Region scope to profile")
    parser.add_argument("--cache", help="Cache configuration")
    parser.add_argument("--compare-cache", action="store_true", help="Compare different cache configurations")
    parser.add_argument("--no-cprofile", action="store_true", help="Disable detailed function profiling")
    
    args = parser.parse_args()
    
    if args.compare_cache:
        profile_setup_methods()
    else:
        profile_model_setup(
            scope=args.scope, 
            cache_type=args.cache,
            enable_cprofile=not args.no_cprofile
        )