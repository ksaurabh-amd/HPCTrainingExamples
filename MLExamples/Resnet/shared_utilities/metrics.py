"""
Performance metrics and monitoring utilities for ResNet workshop.

This module provides consistent performance measurement and analysis
across all workshop versions to ensure fair comparisons.
"""

import torch
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import psutil
from datetime import datetime


@dataclass
class TimingMetrics:
    """Container for timing measurements."""
    batch_times: List[float] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    backward_times: List[float] = field(default_factory=list)
    optimizer_times: List[float] = field(default_factory=list)
    data_loading_times: List[float] = field(default_factory=list)
    
    def add_batch_time(self, time_ms: float):
        """Add batch processing time in milliseconds."""
        self.batch_times.append(time_ms)
    
    def add_forward_time(self, time_ms: float):
        """Add forward pass time in milliseconds."""
        self.forward_times.append(time_ms)
    
    def add_backward_time(self, time_ms: float):
        """Add backward pass time in milliseconds."""
        self.backward_times.append(time_ms)
    
    def add_optimizer_time(self, time_ms: float):
        """Add optimizer step time in milliseconds."""
        self.optimizer_times.append(time_ms)
    
    def add_data_loading_time(self, time_ms: float):
        """Add data loading time in milliseconds."""
        self.data_loading_times.append(time_ms)


@dataclass
class AccuracyMetrics:
    """Container for accuracy measurements."""
    train_accuracies: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    
    def add_train_metrics(self, accuracy: float, loss: float):
        """Add training accuracy and loss."""
        self.train_accuracies.append(accuracy)
        self.train_losses.append(loss)
    
    def add_val_metrics(self, accuracy: float, loss: float):
        """Add validation accuracy and loss."""
        self.val_accuracies.append(accuracy)
        self.val_losses.append(loss)


@dataclass
class MemoryMetrics:
    """Container for memory measurements."""
    peak_memory_mb: float = 0.0
    allocated_memory_mb: List[float] = field(default_factory=list)
    reserved_memory_mb: List[float] = field(default_factory=list)
    cpu_memory_mb: List[float] = field(default_factory=list)
    
    def record_gpu_memory(self):
        """Record current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            self.allocated_memory_mb.append(allocated)
            self.reserved_memory_mb.append(reserved)
            self.peak_memory_mb = max(self.peak_memory_mb, allocated)
    
    def record_cpu_memory(self):
        """Record current CPU memory usage."""
        try:
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            self.cpu_memory_mb.append(memory_mb)
        except:
            pass  # psutil might not be available


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
    
    def get_time_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed_ms


class ThroughputCalculator:
    """Calculate training throughput metrics."""
    
    @staticmethod
    def samples_per_second(batch_size: int, batch_time_ms: float) -> float:
        """Calculate samples per second."""
        if batch_time_ms <= 0:
            return 0.0
        return batch_size / (batch_time_ms / 1000.0)
    
    @staticmethod
    def batches_per_second(batch_time_ms: float) -> float:
        """Calculate batches per second."""
        if batch_time_ms <= 0:
            return 0.0
        return 1000.0 / batch_time_ms
    
    @staticmethod
    def tokens_per_second(batch_size: int, sequence_length: int, batch_time_ms: float) -> float:
        """Calculate tokens per second (for sequence models)."""
        if batch_time_ms <= 0:
            return 0.0
        return (batch_size * sequence_length) / (batch_time_ms / 1000.0)


class ModelProfiler:
    """Comprehensive model performance profiler."""
    
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.timing_metrics = TimingMetrics()
        self.accuracy_metrics = AccuracyMetrics()
        self.memory_metrics = MemoryMetrics()
        self.system_info = self._get_system_info()
        
        # Performance counters
        self.total_samples_processed = 0
        self.total_batches_processed = 0
        self.start_time = None
        self.end_time = None
    
    def start_profiling(self):
        """Start profiling session."""
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def end_profiling(self):
        """End profiling session."""
        self.end_time = time.time()
    
    def record_batch(self, batch_size: int, batch_time_ms: float, 
                    forward_time_ms: float, backward_time_ms: float, 
                    optimizer_time_ms: float, train_acc: float, train_loss: float):
        """Record metrics for a training batch."""
        
        self.timing_metrics.add_batch_time(batch_time_ms)
        self.timing_metrics.add_forward_time(forward_time_ms)
        self.timing_metrics.add_backward_time(backward_time_ms)
        self.timing_metrics.add_optimizer_time(optimizer_time_ms)
        
        self.accuracy_metrics.add_train_metrics(train_acc, train_loss)
        self.memory_metrics.record_gpu_memory()
        self.memory_metrics.record_cpu_memory()
        
        self.total_samples_processed += batch_size
        self.total_batches_processed += 1
    
    def record_validation(self, val_acc: float, val_loss: float):
        """Record validation metrics."""
        self.accuracy_metrics.add_val_metrics(val_acc, val_loss)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        
        if not self.timing_metrics.batch_times:
            return {}
        
        batch_times = np.array(self.timing_metrics.batch_times)
        forward_times = np.array(self.timing_metrics.forward_times) if self.timing_metrics.forward_times else np.array([0])
        backward_times = np.array(self.timing_metrics.backward_times) if self.timing_metrics.backward_times else np.array([0])
        optimizer_times = np.array(self.timing_metrics.optimizer_times) if self.timing_metrics.optimizer_times else np.array([0])
        
        train_accuracies = np.array(self.accuracy_metrics.train_accuracies) if self.accuracy_metrics.train_accuracies else np.array([0])
        train_losses = np.array(self.accuracy_metrics.train_losses) if self.accuracy_metrics.train_losses else np.array([0])
        
        # Calculate throughput
        avg_batch_time_ms = np.mean(batch_times)
        batch_size = self.total_samples_processed // self.total_batches_processed if self.total_batches_processed > 0 else 32
        throughput = ThroughputCalculator.samples_per_second(batch_size, avg_batch_time_ms)
        
        # Calculate total training time
        total_time_s = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        summary = {
            # Timing metrics
            'avg_batch_time_ms': float(avg_batch_time_ms),
            'avg_forward_time_ms': float(np.mean(forward_times)),
            'avg_backward_time_ms': float(np.mean(backward_times)),
            'avg_optimizer_time_ms': float(np.mean(optimizer_times)),
            'std_batch_time_ms': float(np.std(batch_times)),
            
            # Throughput metrics
            'throughput_samples_per_sec': float(throughput),
            'throughput_batches_per_sec': float(ThroughputCalculator.batches_per_second(avg_batch_time_ms)),
            
            # Accuracy metrics
            'final_train_accuracy': float(np.mean(train_accuracies[-5:])) if len(train_accuracies) >= 5 else float(train_accuracies[-1]) if len(train_accuracies) > 0 else 0,
            'final_train_loss': float(np.mean(train_losses[-5:])) if len(train_losses) >= 5 else float(train_losses[-1]) if len(train_losses) > 0 else 0,
            'best_train_accuracy': float(np.max(train_accuracies)) if len(train_accuracies) > 0 else 0,
            'final_val_accuracy': float(self.accuracy_metrics.val_accuracies[-1]) if self.accuracy_metrics.val_accuracies else 0,
            'final_val_loss': float(self.accuracy_metrics.val_losses[-1]) if self.accuracy_metrics.val_losses else 0,
            
            # Memory metrics
            'peak_memory_mb': float(self.memory_metrics.peak_memory_mb),
            'avg_memory_mb': float(np.mean(self.memory_metrics.allocated_memory_mb)) if self.memory_metrics.allocated_memory_mb else 0,
            
            # Summary metrics
            'total_samples_processed': self.total_samples_processed,
            'total_batches_processed': self.total_batches_processed,
            'total_training_time_s': total_time_s,
            
            # Efficiency metrics
            'forward_backward_ratio': float(np.mean(forward_times) / np.mean(backward_times)) if np.mean(backward_times) > 0 else 0,
            'compute_efficiency_pct': float((np.mean(forward_times) + np.mean(backward_times)) / avg_batch_time_ms * 100) if avg_batch_time_ms > 0 else 0,
        }
        
        return summary
    
    def save_results(self, output_dir: str, filename: Optional[str] = None):
        """Save profiling results to JSON file."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_profile_{self.model_name}_{self.version}_{timestamp}.json"
        
        results = {
            'metadata': {
                'model_name': self.model_name,
                'version': self.version,
                'timestamp': datetime.now().isoformat(),
                'system_info': self.system_info
            },
            'performance_summary': self.get_summary_stats(),
            'detailed_metrics': {
                'timing': {
                    'batch_times_ms': self.timing_metrics.batch_times,
                    'forward_times_ms': self.timing_metrics.forward_times,
                    'backward_times_ms': self.timing_metrics.backward_times,
                    'optimizer_times_ms': self.timing_metrics.optimizer_times,
                },
                'accuracy': {
                    'train_accuracies': self.accuracy_metrics.train_accuracies,
                    'train_losses': self.accuracy_metrics.train_losses,
                    'val_accuracies': self.accuracy_metrics.val_accuracies,
                    'val_losses': self.accuracy_metrics.val_losses,
                },
                'memory': {
                    'allocated_memory_mb': self.memory_metrics.allocated_memory_mb,
                    'peak_memory_mb': self.memory_metrics.peak_memory_mb,
                }
            }
        }
        
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Performance results saved to: {output_file}")
        return output_file
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        
        info = {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'cuda_version': torch.version.cuda,
            })
        
        try:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1e9,
            })
        except:
            pass
        
        return info


def compare_performance_results(result_files: List[str]) -> Dict[str, Any]:
    """Compare performance results across multiple versions."""
    
    results = {}
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            version = data['metadata']['version']
            results[version] = data['performance_summary']
    
    # Calculate relative improvements
    if 'v1_baseline' in results:
        baseline = results['v1_baseline']
        for version, metrics in results.items():
            if version != 'v1_baseline':
                metrics['speedup_vs_baseline'] = metrics['throughput_samples_per_sec'] / baseline['throughput_samples_per_sec']
                metrics['memory_reduction_vs_baseline'] = (baseline['peak_memory_mb'] - metrics['peak_memory_mb']) / baseline['peak_memory_mb']
    
    return results


def print_performance_summary(summary: Dict[str, Any], title: str = "Performance Summary"):
    """Print formatted performance summary."""
    
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Timing metrics
    print(f"Timing Metrics:")
    print(f"  Average batch time: {summary.get('avg_batch_time_ms', 0):.2f} ms")
    print(f"  Average forward time: {summary.get('avg_forward_time_ms', 0):.2f} ms")
    print(f"  Average backward time: {summary.get('avg_backward_time_ms', 0):.2f} ms")
    print(f"  Average optimizer time: {summary.get('avg_optimizer_time_ms', 0):.2f} ms")
    
    # Throughput metrics
    print(f"\nThroughput Metrics:")
    print(f"  Samples per second: {summary.get('throughput_samples_per_sec', 0):.1f}")
    print(f"  Batches per second: {summary.get('throughput_batches_per_sec', 0):.2f}")
    
    # Accuracy metrics
    print(f"\nAccuracy Metrics:")
    print(f"  Final training accuracy: {summary.get('final_train_accuracy', 0):.2f}%")
    print(f"  Final training loss: {summary.get('final_train_loss', 0):.4f}")
    if summary.get('final_val_accuracy', 0) > 0:
        print(f"  Final validation accuracy: {summary.get('final_val_accuracy', 0):.2f}%")
    
    # Memory metrics
    print(f"\nMemory Metrics:")
    print(f"  Peak memory usage: {summary.get('peak_memory_mb', 0):.1f} MB")
    print(f"  Average memory usage: {summary.get('avg_memory_mb', 0):.1f} MB")
    
    # Efficiency metrics
    print(f"\nEfficiency Metrics:")
    print(f"  Compute efficiency: {summary.get('compute_efficiency_pct', 0):.1f}%")
    print(f"  Forward/Backward ratio: {summary.get('forward_backward_ratio', 0):.2f}")
    
    # Speedup metrics (if available)
    if 'speedup_vs_baseline' in summary:
        print(f"\nSpeedup vs Baseline:")
        print(f"  Throughput speedup: {summary['speedup_vs_baseline']:.2f}x")
        print(f"  Memory reduction: {summary.get('memory_reduction_vs_baseline', 0)*100:.1f}%")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test performance metrics
    print("Testing performance metrics...")
    
    profiler = ModelProfiler("resnet18", "v1_baseline")
    profiler.start_profiling()
    
    # Simulate some training batches
    for i in range(10):
        profiler.record_batch(
            batch_size=32,
            batch_time_ms=10.0 + i * 0.5,
            forward_time_ms=4.0 + i * 0.2,
            backward_time_ms=5.0 + i * 0.2,
            optimizer_time_ms=1.0 + i * 0.1,
            train_acc=70.0 + i * 2.0,
            train_loss=2.0 - i * 0.1
        )
    
    profiler.end_profiling()
    
    # Get and print summary
    summary = profiler.get_summary_stats()
    print_performance_summary(summary, "Test Performance Summary")
    
    print("Performance metrics test completed!")