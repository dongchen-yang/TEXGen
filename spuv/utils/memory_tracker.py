"""
Memory tracking utilities for debugging OOM issues.
"""

import torch
import gc
from typing import Optional
import spuv


class MemoryTracker:
    """Track and log GPU memory usage at different points in training."""
    
    def __init__(self, enabled: bool = True, log_interval: int = 1):
        self.enabled = enabled
        self.log_interval = log_interval
        self.last_log_step = -1
        self.baseline_allocated = 0
        self.baseline_reserved = 0
        
    def log_memory(self, tag: str, step: Optional[int] = None, force: bool = False):
        """Log current GPU memory usage with a descriptive tag."""
        if not self.enabled:
            return
        
        # Check if we should log based on interval
        if step is not None and not force:
            if step == self.last_log_step:
                return
            if step % self.log_interval != 0:
                return
            self.last_log_step = step
        
        if not torch.cuda.is_available():
            return
        
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**3  # GB
        
        # Get free memory from device properties
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free_memory = total_memory - allocated
        
        # Calculate change from baseline if set
        if self.baseline_allocated > 0:
            delta_allocated = allocated - self.baseline_allocated
            delta_reserved = reserved - self.baseline_reserved
            spuv.info(
                f"[MEMORY] {tag:40s} | "
                f"Allocated: {allocated:6.2f} GB (Δ{delta_allocated:+6.2f}) | "
                f"Reserved: {reserved:6.2f} GB (Δ{delta_reserved:+6.2f}) | "
                f"Free: {free_memory:6.2f} GB | "
                f"Peak: {max_allocated:6.2f} GB"
            )
        else:
            spuv.info(
                f"[MEMORY] {tag:40s} | "
                f"Allocated: {allocated:6.2f} GB | "
                f"Reserved: {reserved:6.2f} GB | "
                f"Free: {free_memory:6.2f} GB | "
                f"Peak: {max_allocated:6.2f} GB"
            )
    
    def set_baseline(self):
        """Set current memory usage as baseline for delta calculations."""
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()
        self.baseline_allocated = torch.cuda.memory_allocated(device) / 1024**3
        self.baseline_reserved = torch.cuda.memory_reserved(device) / 1024**3
        spuv.info(f"[MEMORY] Baseline set: {self.baseline_allocated:.2f} GB allocated, {self.baseline_reserved:.2f} GB reserved")
    
    def reset_peak(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def cleanup(self, tag: str = "cleanup", log: bool = True):
        """Run garbage collection and clear CUDA cache."""
        if log and self.enabled:
            self.log_memory(f"Before {tag}", force=True)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if log and self.enabled:
            self.log_memory(f"After {tag}", force=True)


# Global tracker instance
_global_tracker: Optional[MemoryTracker] = None


def get_tracker() -> MemoryTracker:
    """Get or create the global memory tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker()
    return _global_tracker


def init_tracker(enabled: bool = True, log_interval: int = 1):
    """Initialize the global memory tracker."""
    global _global_tracker
    _global_tracker = MemoryTracker(enabled=enabled, log_interval=log_interval)
    return _global_tracker


def log_memory(tag: str, step: Optional[int] = None, force: bool = False):
    """Convenience function to log memory using global tracker."""
    get_tracker().log_memory(tag, step, force)


def set_baseline():
    """Convenience function to set baseline using global tracker."""
    get_tracker().set_baseline()


def cleanup(tag: str = "cleanup", log: bool = True):
    """Convenience function to cleanup using global tracker."""
    get_tracker().cleanup(tag, log)
