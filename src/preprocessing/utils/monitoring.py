"""
Performance monitoring and system metrics utilities.

This module provides comprehensive monitoring capabilities for tracking
performance, memory usage, and system metrics during pipeline execution.
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import psutil


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    execution_time: float
    stage_name: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking system resources and execution times.

    This class provides comprehensive monitoring of CPU, memory, disk usage,
    and execution times for different pipeline stages.
    """

    def __init__(self, log_file: Optional[str] = None, interval: float = 1.0):
        """
        Initialize PerformanceMonitor.

        Args:
            log_file: Path to log file for metrics
            interval: Monitoring interval in seconds
        """
        self.log_file = log_file
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Ensure log directory exists
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5.0)
            self.logger.info("Performance monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self.stop_monitoring.wait(self.interval):
            try:
                metrics = self._collect_metrics("background")
                self.metrics.append(metrics)

                if self.log_file:
                    self._log_metrics(metrics)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _collect_metrics(self, stage_name: str) -> PerformanceMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            execution_time=0.0,  # Will be set by context manager
            stage_name=stage_name,
        )

    def _log_metrics(self, metrics: PerformanceMetrics) -> None:
        """Log metrics to file."""
        if not self.log_file:
            return

        log_entry = {
            "timestamp": metrics.timestamp.isoformat(),
            "stage": metrics.stage_name,
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_mb": metrics.memory_used_mb,
            "memory_available_mb": metrics.memory_available_mb,
            "disk_usage_percent": metrics.disk_usage_percent,
            "execution_time": metrics.execution_time,
            "additional_metrics": metrics.additional_metrics,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    @contextmanager
    def monitor_stage(
        self, stage_name: str, additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for monitoring a pipeline stage.

        Args:
            stage_name: Name of the stage being monitored
            additional_metrics: Additional metrics to track
        """
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            end_metrics = self._collect_metrics(stage_name)
            end_metrics.execution_time = execution_time
            end_metrics.additional_metrics = additional_metrics or {}

            self.metrics.append(end_metrics)

            if self.log_file:
                self._log_metrics(end_metrics)

            self.logger.info(
                f"Stage '{stage_name}' completed in {execution_time:.2f}s "
                f"(CPU: {end_metrics.cpu_percent:.1f}%, "
                f"Memory: {end_metrics.memory_percent:.1f}%)"
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {}

        df = pd.DataFrame(
            [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "execution_time": m.execution_time,
                    "stage_name": m.stage_name,
                }
                for m in self.metrics
            ]
        )

        return {
            "total_metrics": len(self.metrics),
            "monitoring_duration": (
                df["timestamp"].max() - df["timestamp"].min()
            ).total_seconds(),
            "avg_cpu_percent": df["cpu_percent"].mean(),
            "max_cpu_percent": df["cpu_percent"].max(),
            "avg_memory_percent": df["memory_percent"].mean(),
            "max_memory_percent": df["memory_percent"].max(),
            "avg_memory_used_mb": df["memory_used_mb"].mean(),
            "max_memory_used_mb": df["memory_used_mb"].max(),
            "total_execution_time": df["execution_time"].sum(),
            "stage_breakdown": df.groupby("stage_name")["execution_time"]
            .sum()
            .to_dict(),
        }

    def save_report(self, output_path: str) -> None:
        """Save detailed performance report."""
        summary = self.get_summary()

        report = {
            "summary": summary,
            "detailed_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "stage_name": m.stage_name,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "execution_time": m.execution_time,
                    "additional_metrics": m.additional_metrics,
                }
                for m in self.metrics
            ],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Performance report saved to {output_path}")


class MemoryProfiler:
    """
    Memory profiling utility for tracking memory usage of specific functions.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize MemoryProfiler.

        Args:
            log_file: Path to log file for memory profiles
        """
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.profiles: List[Dict[str, Any]] = []

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def profile_function(
        self, function_name: str, additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for profiling memory usage of a function.

        Args:
            function_name: Name of the function being profiled
            additional_info: Additional information to log
        """
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory

            profile = {
                "timestamp": datetime.now().isoformat(),
                "function_name": function_name,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
                "memory_delta_mb": memory_delta,
                "execution_time": execution_time,
                "additional_info": additional_info or {},
            }

            self.profiles.append(profile)

            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(profile) + "\n")

            self.logger.info(
                f"Function '{function_name}' memory profile: "
                f"start={start_memory:.1f}MB, end={end_memory:.1f}MB, "
                f"delta={memory_delta:+.1f}MB, time={execution_time:.2f}s"
            )

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory profiling summary."""
        if not self.profiles:
            return {}

        df = pd.DataFrame(self.profiles)

        return {
            "total_functions_profiled": len(self.profiles),
            "total_memory_used_mb": df["end_memory_mb"].max(),
            "max_memory_delta_mb": df["memory_delta_mb"].max(),
            "total_execution_time": df["execution_time"].sum(),
            "function_breakdown": df.groupby("function_name")
            .agg(
                {
                    "memory_delta_mb": ["mean", "max", "sum"],
                    "execution_time": ["mean", "max", "sum"],
                }
            )
            .to_dict(),
        }


class PipelineMonitor:
    """
    High-level pipeline monitoring utility that combines performance and memory monitoring.
    """

    def __init__(self, output_dir: str = "logs/monitoring"):
        """
        Initialize PipelineMonitor.

        Args:
            output_dir: Output directory for monitoring logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.performance_monitor = PerformanceMonitor(
            log_file=self.output_dir / "performance.log"
        )
        self.memory_profiler = MemoryProfiler(log_file=self.output_dir / "memory.log")

        self.logger = logging.getLogger(__name__)

    def start_monitoring(self) -> None:
        """Start all monitoring components."""
        self.performance_monitor.start_monitoring()
        self.logger.info("Pipeline monitoring started")

    def stop_monitoring(self) -> None:
        """Stop all monitoring components."""
        self.performance_monitor.stop_monitoring()
        self.logger.info("Pipeline monitoring stopped")

    @contextmanager
    def monitor_pipeline_stage(
        self, stage_name: str, additional_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for monitoring a pipeline stage with both performance and memory tracking.

        Args:
            stage_name: Name of the pipeline stage
            additional_metrics: Additional metrics to track
        """
        with self.performance_monitor.monitor_stage(stage_name, additional_metrics):
            with self.memory_profiler.profile_function(stage_name, additional_metrics):
                yield

    def generate_report(self, report_name: str = None) -> str:
        """Generate comprehensive monitoring report."""
        if report_name is None:
            report_name = (
                f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        report_path = self.output_dir / report_name

        # Generate performance report
        performance_report_path = self.output_dir / "performance_report.json"
        self.performance_monitor.save_report(str(performance_report_path))

        # Get summaries
        performance_summary = self.performance_monitor.get_summary()
        memory_summary = self.memory_profiler.get_memory_summary()

        # Combine reports
        combined_report = {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": performance_summary,
            "memory_summary": memory_summary,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage": psutil.disk_usage("/").percent,
            },
        }

        with open(report_path, "w") as f:
            json.dump(combined_report, f, indent=2, default=str)

        self.logger.info(f"Comprehensive monitoring report saved to {report_path}")
        return str(report_path)


# Convenience functions for easy usage
def monitor_function(func: Callable) -> Callable:
    """
    Decorator for monitoring function performance and memory usage.

    Args:
        func: Function to monitor

    Returns:
        Wrapped function with monitoring
    """

    def wrapper(*args, **kwargs):
        monitor = PipelineMonitor()
        with monitor.memory_profiler.profile_function(func.__name__):
            with monitor.performance_monitor.monitor_stage(func.__name__):
                return func(*args, **kwargs)

    return wrapper


def get_system_info() -> Dict[str, Any]:
    """Get current system information."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_total_gb": memory.total / (1024**3),
        "memory_available_gb": memory.available / (1024**3),
        "memory_percent": memory.percent,
        "disk_total_gb": disk.total / (1024**3),
        "disk_free_gb": disk.free / (1024**3),
        "disk_percent": disk.percent,
    }
