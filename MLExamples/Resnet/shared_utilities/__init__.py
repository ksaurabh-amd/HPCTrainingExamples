"""
Shared utilities for ResNet workshop.

This package provides common functionality used across all workshop versions
to ensure consistent performance measurement and analysis.
"""

__version__ = "1.0.0"

# Import key classes and functions for easy access
from .datasets import (
    create_data_loaders,
    get_dataset_info,
    DatasetConfig,
    SubsetDataLoader,
    get_quick_loaders
)

from .metrics import (
    ModelProfiler,
    PerformanceTimer,
    ThroughputCalculator,
    TimingMetrics,
    AccuracyMetrics,
    MemoryMetrics,
    print_performance_summary,
    compare_performance_results
)

__all__ = [
    # Dataset utilities
    'create_data_loaders',
    'get_dataset_info', 
    'DatasetConfig',
    'SubsetDataLoader',
    'get_quick_loaders',
    
    # Performance metrics
    'ModelProfiler',
    'PerformanceTimer',
    'ThroughputCalculator',
    'TimingMetrics',
    'AccuracyMetrics', 
    'MemoryMetrics',
    'print_performance_summary',
    'compare_performance_results',
]