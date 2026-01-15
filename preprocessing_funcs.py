# universal functions which can be edited to condition signals for fiducial point extraction and quality checks
from .signal_processing_utils import apply_butterworth_filter

def preprocess_earbud(data, fs=1000):
    data = apply_butterworth_filter(data, .5, "high", fs=fs) 
    data = apply_butterworth_filter(data, 15, "low", fs=fs) 
    return data

def preprocess_ppg(data, fs=1000):
    data = apply_butterworth_filter(data, .5, "high", fs=fs) 
    data = apply_butterworth_filter(data, 15, "low", fs=fs) 
    return data

def preprocess_ecg(data, fs=1000):
    data = apply_butterworth_filter(data, 0.5, "high", fs=fs)
    data = apply_butterworth_filter(data, 40, "low", fs=fs)
    return data

def preprocess_accelerometer(data, fs=1000):
    data = apply_butterworth_filter(data, .5, "high", fs=fs) 
    return data

def preprocess_stethoscope(data, fs=1000):
    data = apply_butterworth_filter(data, 40, "high", fs=fs) 
    data = apply_butterworth_filter(data, 100, "low", fs=fs) 
    return data

def preprocess_respiration_belt(data, fs=1000):
    data = apply_butterworth_filter(data, 0.1, "high", fs=fs) 
    data = apply_butterworth_filter(data, 0.5, "low", fs=fs) 
    return data