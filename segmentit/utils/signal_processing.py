"""Utility functions for signal processing."""

from typing import Tuple, Optional
import numpy as np
from numba import jit

@jit(nopython=True)
def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to zero mean and unit standard deviation.
    
    Args:
        signal: Input signal
        
    Returns:
        Normalized signal
    """
    mean = np.mean(signal)
    std = np.std(signal)
    if std > 0:
        return (signal - mean) / std
    return signal - mean

@jit(nopython=True)
def median_filter(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply median filter to signal.
    
    Args:
        signal: Input signal
        window_size: Size of the median filter window
        
    Returns:
        Filtered signal
    """
    result = np.zeros_like(signal)
    half_window = window_size // 2
    signal_len = len(signal)
    
    # Handle edges separately
    for i in range(half_window):
        window = signal[0:i+half_window+1]
        result[i] = np.median(window)
        
    # Main signal
    for i in range(half_window, signal_len - half_window):
        window = signal[i-half_window:i+half_window+1]
        result[i] = np.median(window)
        
    # Handle edges separately
    for i in range(signal_len - half_window, signal_len):
        window = signal[i-half_window:signal_len]
        result[i] = np.median(window)
        
    return result

@jit(nopython=True)
def detect_peaks(signal: np.ndarray, threshold: float, min_distance: int = 10) -> np.ndarray:
    """Detect peaks in signal.
    
    Args:
        signal: Input signal
        threshold: Threshold for peak detection
        min_distance: Minimum distance between peaks
        
    Returns:
        Array of peak indices
    """
    # Find indices where signal exceeds threshold
    peak_indices = []
    signal_len = len(signal)
    
    for i in range(1, signal_len - 1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            peak_indices.append(i)
    
    # Apply minimum distance constraint
    if not peak_indices:
        return np.array([], dtype=np.int32)
        
    filtered_peaks = [peak_indices[0]]
    last_peak = peak_indices[0]
    
    for peak in peak_indices[1:]:
        if peak - last_peak >= min_distance:
            filtered_peaks.append(peak)
            last_peak = peak
            
    return np.array(filtered_peaks, dtype=np.int32)

@jit(nopython=True)
def running_stats(signal: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate running mean and standard deviation.
    
    Args:
        signal: Input signal
        window_size: Window size
        
    Returns:
        Tuple of (running_mean, running_std)
    """
    signal_len = len(signal)
    running_mean = np.zeros(signal_len)
    running_std = np.zeros(signal_len)
    
    for i in range(signal_len):
        start = max(0, i - window_size + 1)
        window = signal[start:i+1]
        running_mean[i] = np.mean(window)
        running_std[i] = np.std(window)
        
    return running_mean, running_std

def rescale_signal(signal: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """Rescale signal to a target range.
    
    Args:
        signal: Input signal
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        Rescaled signal
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    
    if signal_max == signal_min:
        return np.ones_like(signal) * (target_min + target_max) / 2
        
    scaled = (signal - signal_min) / (signal_max - signal_min) * (target_max - target_min) + target_min
    return scaled