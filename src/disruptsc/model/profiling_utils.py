"""
Profiling utilities for DisruptSC model performance analysis.

This module provides decorators and utilities for timing critical model operations.
"""

import time
import logging
import functools
from typing import Dict, Any, Callable
from collections import defaultdict


class ModelProfiler:
    """Singleton class to collect timing data across model operations."""
    
    _instance = None
    _timings = defaultdict(list)
    _enabled = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def enable(cls):
        """Enable profiling."""
        cls._enabled = True
        cls._timings.clear()
        logging.info("Model profiling enabled")
    
    @classmethod
    def disable(cls):
        """Disable profiling."""
        cls._enabled = False
        logging.info("Model profiling disabled")
    
    @classmethod
    def is_enabled(cls):
        """Check if profiling is enabled."""
        return cls._enabled
    
    @classmethod
    def record_timing(cls, method_name: str, duration: float):
        """Record timing for a method."""
        if cls._enabled:
            cls._timings[method_name].append(duration)
    
    @classmethod
    def get_summary(cls) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all recorded timings."""
        summary = {}
        for method_name, durations in cls._timings.items():
            if durations:
                summary[method_name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return summary
    
    @classmethod
    def print_summary(cls):
        """Print formatted summary of timing data."""
        summary = cls.get_summary()
        if not summary:
            print("No profiling data available")
            return
        
        print("\n" + "="*80)
        print("🔍 MODEL PROFILING SUMMARY")
        print("="*80)
        print(f"{'Method':<40} {'Count':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10}")
        print("-" * 80)
        
        # Sort by total time
        sorted_methods = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for method_name, stats in sorted_methods:
            print(f"{method_name:<40} {stats['count']:<8} {stats['total']:<10.3f} "
                  f"{stats['average']:<10.3f} {stats['min']:<10.3f} {stats['max']:<10.3f}")
        
        print("="*80)
        total_time = sum(stats['total'] for stats in summary.values())
        print(f"Total profiled time: {total_time:.3f}s")
        print("="*80)
    
    @classmethod
    def clear(cls):
        """Clear all timing data."""
        cls._timings.clear()


def profile_method(func: Callable) -> Callable:
    """
    Decorator to profile method execution time.
    
    Usage:
        @profile_method
        def some_method(self):
            # method implementation
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ModelProfiler.is_enabled():
            return func(*args, **kwargs)
        
        # Get method name with class if available
        method_name = func.__name__
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            method_name = f"{class_name}.{method_name}"
        
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start_time
            ModelProfiler.record_timing(method_name, duration)
            logging.debug(f"⏱️  {method_name}: {duration:.3f}s")
    
    return wrapper


def profile_function(name: str = None):
    """
    Decorator to profile function execution time with custom name.
    
    Usage:
        @profile_function("Custom Function Name")
        def some_function():
            # function implementation
    """
    def decorator(func: Callable) -> Callable:
        function_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ModelProfiler.is_enabled():
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                ModelProfiler.record_timing(function_name, duration)
                logging.debug(f"⏱️  {function_name}: {duration:.3f}s")
        
        return wrapper
    return decorator


class ProfiledModel:
    """Context manager to enable profiling for model operations."""
    
    def __init__(self, clear_previous: bool = True):
        self.clear_previous = clear_previous
    
    def __enter__(self):
        if self.clear_previous:
            ModelProfiler.clear()
        ModelProfiler.enable()
        return ModelProfiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        ModelProfiler.disable()


# Convenience functions
def enable_profiling():
    """Enable model profiling."""
    ModelProfiler.enable()


def disable_profiling():
    """Disable model profiling."""
    ModelProfiler.disable()


def print_profiling_summary():
    """Print profiling summary."""
    ModelProfiler.print_summary()


def clear_profiling_data():
    """Clear all profiling data."""
    ModelProfiler.clear()