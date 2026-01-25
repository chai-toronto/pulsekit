from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
from scipy.signal import find_peaks
import numpy as np

class Stethoscope(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 40, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 100, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        pcg_segment = preprocessed[window_start + 10 : window_start + 150]
        
        if len(pcg_segment) == 0:
             return {}

        pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
        pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
        
        if len(pcg_peaks) > 0:
            return {f"{self.name}_s1": window_start + 10 + pcg_peaks[0]}
        
        return {}

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        window = preprocessed[window_start:window_end]
        # Basic check: is there signal variation?
        return (np.std(window) > 0.001)
    
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