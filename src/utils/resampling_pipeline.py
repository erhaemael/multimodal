from typing import Dict
import numpy as np
from scipy.signal import gaussian, convolve
from scipy.interpolate import interp1d
from src.data.WESAD.constants import TARGET_SR 

def gaussian_smoothing(signal, window_size=40, sigma=400):
    """
    Apply Gaussian smoothing to the input signal.
    
    Parameters:
    - signal: 1D array, the signal to smooth.
    - window_size: int, the size of the Gaussian window.
    - sigma: float, the standard deviation of the Gaussian kernel.
    
    Returns:
    - smoothed_signal: 1D array, the signal after smoothing.
    """
    gauss_win = gaussian(window_size, std=sigma) 
    gauss_win /= np.sum(gauss_win)                
    return convolve(signal, gauss_win, mode='same') 

def upsample_interp(signal, original_sr, target_sr):
    """
    Perform linear interpolation to upsample the input signal from original sampling rate to target sampling rate.
    
    Parameters:
    - signal: 1D array, the original signal.
    - original_sr: int, original sampling rate of the signal.
    - target_sr: int, desired target sampling rate.
    
    Returns:
    - upsampled_signal: 1D array, the signal after upsampling.
    """
    duration = len(signal) / original_sr                   
    original_time = np.linspace(0, duration, len(signal))  
    target_time = np.linspace(0, duration, int(duration * target_sr))  
    interpolator = interp1d(original_time, signal, kind='linear', fill_value="extrapolate") 
    return interpolator(target_time)                       

def resample_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Resample physiological signals to a common target sampling rate.
    
    Process details:
    - BVP signal is already at target frequency (64 Hz) and used as is.
    - EDA signal is upsampled from 4 Hz to target rate, then smoothed using Gaussian kernel.
    - TEMP signal is upsampled from 4 Hz to target rate, then clipped to a realistic temperature range.
    - All signals are truncated to the shortest length among them to maintain alignment.
    
    Parameters:
    - signals: dictionary containing raw signals ('BVP', 'EDA', 'TEMP') as numpy arrays.
    
    Returns:
    - resampled_signals: dictionary containing processed signals at target sampling rate.
    """
    bvp_signal = signals['BVP'].flatten()  

    eda_signal = signals['EDA'].flatten()
    eda_upsampled = upsample_interp(eda_signal, original_sr=4, target_sr=TARGET_SR)
    eda_smoothed = gaussian_smoothing(eda_upsampled, window_size=40, sigma=400)

    temp_signal = signals['TEMP'].flatten()
    temp_upsampled = upsample_interp(temp_signal, original_sr=4, target_sr=TARGET_SR)
    temp_cleaned = np.clip(temp_upsampled, 20, 50)  # Ensure temperature values are within a realistic range (20°C to 50°C)

    # Determine the minimum length to synchronize all signals
    min_len = min(len(bvp_signal), len(eda_smoothed), len(temp_cleaned))

    # Return the resampled and synchronized signals
    return {
        "BVP": bvp_signal[:min_len],
        "EDA": eda_smoothed[:min_len],
        "TEMP": temp_cleaned[:min_len],
    }
