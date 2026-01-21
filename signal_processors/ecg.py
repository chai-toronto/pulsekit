from .signal_processing_utils import apply_butterworth_filter
import numpy as np
from scipy.stats import kurtosis
from typing import Dict, Any
from .base import PhysiologicalSignal
from scipy.signal import find_peaks

class ECG(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 40, "low", fs=self.sampling_rate)

    def find_peaks(self):
        ecg_signal = self.data
        detection_signal = apply_butterworth_filter(ecg_signal, 5, "high", fs=self.sampling_rate)
        detection_signal = apply_butterworth_filter(detection_signal, 15, "low", fs=self.sampling_rate)
        squared_signal = detection_signal ** 2
        height_req = np.mean(squared_signal)
        rough_peaks, _ = find_peaks(squared_signal, distance=400, height=height_req)
        true_peaks = []
        search_window = int(0.05 * self.sampling_rate) # +/- 50ms window 
        
        for p in rough_peaks:
            start = max(0, p - search_window)
            end = min(len(ecg_signal), p + search_window)
            search_slice = ecg_signal[start:end]
            local_max_idx = np.argmax(search_slice)
            true_peak_idx = start + local_max_idx
            true_peaks.append(true_peak_idx)
            
        return np.array(true_peaks)

    def extract_cardiac_cycle_fiducials(self, preprocessed, beat_start: int, beat_end: int):
        return {f"{self.name}_R_peak": beat_start}

    def check_quality_cardiac_cycle(self, preprocessed, beat_start: int, beat_end: int, fiducials: dict):
        window = preprocessed[beat_start:beat_end]
        if len(window) == 0: return False

        try:
            tests = [
                (np.std(window) > 0.001), 
                (np.max(window[:100]) + np.max(window[-100:])) / 2 > np.mean(window), 
                (kurtosis(window, fisher=False) > 3.5)
            ]
            return all(tests)
        except Exception:
            return False
        
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        # basic placeholder to ensure window has data that is not 60Hz line noise
        sig = preprocessed[window_start:window_end]
        if np.ptp(sig) < 0.001: 
            return False 
        fft_mags = np.abs(np.fft.rfft(sig))
        hz_60_bin = int(60 * len(sig) / self.sampling_rate)
        if fft_mags[hz_60_bin] > (np.mean(fft_mags) * 10):
            return False
        return True
    
    def extract_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        # basic placeholder to return stats in the window
        window_data = preprocessed[window_start:window_end]
        return {
            f"{self.name}_mean": np.mean(window_data),
            f"{self.name}_std": np.std(window_data),
            f"{self.name}_max": np.max(window_data),
            f"{self.name}_min": np.min(window_data)
        }