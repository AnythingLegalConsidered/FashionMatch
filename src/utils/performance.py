"""Performance monitoring and profiling utilities."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import psutil

from src.utils import get_logger

logger = get_logger(__name__)

# Try to import torch for GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    
    operation: str
    duration_seconds: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    items_processed: int = 0
    throughput: float = 0.0  # items/second


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, list[PerformanceMetrics]] = {}
        self._enabled = False
    
    def enable(self) -> None:
        """Enable performance monitoring."""
        self._enabled = True
        logger.info("Performance monitoring enabled")
    
    def disable(self) -> None:
        """Disable performance monitoring."""
        self._enabled = False
        logger.info("Performance monitoring disabled")
    
    @contextmanager
    def measure(self, operation: str, items_count: int = 0):
        """Context manager for measuring operation performance.
        
        Args:
            operation: Name of the operation being measured
            items_count: Number of items processed (for throughput calculation)
        
        Example:
            >>> monitor = get_performance_monitor()
            >>> with monitor.measure("encode_batch", items_count=32):
            ...     encoder.encode(images)
        """
        if not self._enabled:
            yield
            return
        
        # Measure start state
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_gpu_memory = None
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        try:
            yield
        finally:
            # Measure end state
            duration = time.time() - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            gpu_memory_delta = None
            if start_gpu_memory is not None:
                torch.cuda.synchronize()
                end_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            throughput = items_count / duration if duration > 0 and items_count > 0 else 0.0
            
            metric = PerformanceMetrics(
                operation=operation,
                duration_seconds=duration,
                memory_mb=memory_delta,
                gpu_memory_mb=gpu_memory_delta,
                items_processed=items_count,
                throughput=throughput
            )
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(metric)
            
            # Log metric
            log_msg = (
                f"Performance [{operation}]: "
                f"duration={duration:.3f}s, "
                f"memory={memory_delta:+.1f}MB"
            )
            if items_count > 0:
                log_msg += f", throughput={throughput:.1f} items/s"
            if gpu_memory_delta is not None:
                log_msg += f", gpu_memory={gpu_memory_delta:+.1f}MB"
            
            logger.debug(log_msg)
    
    def get_summary(self, operation: Optional[str] = None) -> Dict:
        """Get performance summary statistics.
        
        Args:
            operation: Optional specific operation to summarize
        
        Returns:
            Dictionary with summary statistics
        """
        if operation:
            metrics = self.metrics.get(operation, [])
        else:
            metrics = [m for ms in self.metrics.values() for m in ms]
        
        if not metrics:
            return {}
        
        total_duration = sum(m.duration_seconds for m in metrics)
        avg_duration = total_duration / len(metrics)
        avg_memory = sum(m.memory_mb for m in metrics) / len(metrics)
        total_items = sum(m.items_processed for m in metrics)
        
        # Calculate average throughput (only for metrics with items)
        metrics_with_items = [m for m in metrics if m.items_processed > 0]
        avg_throughput = 0.0
        if metrics_with_items:
            avg_throughput = sum(m.throughput for m in metrics_with_items) / len(metrics_with_items)
        
        summary = {
            'count': len(metrics),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'min_duration': min(m.duration_seconds for m in metrics),
            'max_duration': max(m.duration_seconds for m in metrics),
            'avg_memory_mb': avg_memory,
            'total_items': total_items,
            'avg_throughput': avg_throughput
        }
        
        # Add GPU stats if available
        gpu_metrics = [m for m in metrics if m.gpu_memory_mb is not None]
        if gpu_metrics:
            summary['avg_gpu_memory_mb'] = sum(m.gpu_memory_mb for m in gpu_metrics) / len(gpu_metrics)
        
        return summary
    
    def print_report(self) -> None:
        """Print formatted performance report."""
        if not self.metrics:
            logger.info("No performance metrics recorded")
            return
        
        print("\n" + "=" * 80)
        print("PERFORMANCE REPORT")
        print("=" * 80)
        
        for operation, metrics in sorted(self.metrics.items()):
            summary = self.get_summary(operation)
            print(f"\n{operation}:")
            print(f"  Calls:            {summary['count']}")
            print(f"  Total Duration:   {summary['total_duration']:.2f}s")
            print(f"  Avg Duration:     {summary['avg_duration']:.3f}s")
            print(f"  Min Duration:     {summary['min_duration']:.3f}s")
            print(f"  Max Duration:     {summary['max_duration']:.3f}s")
            print(f"  Avg Memory:       {summary['avg_memory_mb']:+.1f} MB")
            
            if summary['total_items'] > 0:
                print(f"  Total Items:      {summary['total_items']}")
                print(f"  Avg Throughput:   {summary['avg_throughput']:.1f} items/s")
            
            if 'avg_gpu_memory_mb' in summary:
                print(f"  Avg GPU Memory:   {summary['avg_gpu_memory_mb']:+.1f} MB")
        
        print("=" * 80 + "\n")
    
    def clear(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()
        logger.debug("Performance metrics cleared")
    
    def export_to_dict(self) -> Dict:
        """Export all metrics to dictionary format.
        
        Returns:
            Dictionary with all recorded metrics
        """
        export = {}
        for operation, metrics in self.metrics.items():
            export[operation] = [
                {
                    'duration_seconds': m.duration_seconds,
                    'memory_mb': m.memory_mb,
                    'gpu_memory_mb': m.gpu_memory_mb,
                    'items_processed': m.items_processed,
                    'throughput': m.throughput
                }
                for m in metrics
            ]
        return export


# Global monitor instance
_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    return _monitor
