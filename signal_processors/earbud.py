from .base import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
import numpy as np

class Earbud(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 15, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):

        theoretical_min = int(0.1 * 0.3 * self.sampling_rate) - 10 
        theoretical_max = 500 
        
        win_start = window_start + theoretical_min
        win_end = min(window_start + theoretical_max, len(preprocessed))
        arrival_window = preprocessed[win_start : win_end]
        
        if len(arrival_window) == 0:
            return {}

        a_peak_global = np.argmax(arrival_window) + win_start
        
        segment_start_idx = max(0, window_start - 200)
        segment_end_idx = min(len(preprocessed), window_end + 200)
        segment = preprocessed[segment_start_idx : segment_end_idx]
        
        anchor_rel = a_peak_global - segment_start_idx
        points = extract_fiducial_points(segment, c_peak=anchor_rel)
        
        return {
            f"{self.name}_a": points[0] + segment_start_idx,
            f"{self.name}_b": points[1] + segment_start_idx,
            f"{self.name}_c": points[2] + segment_start_idx,
            f"{self.name}_d": points[3] + segment_start_idx,
            f"{self.name}_e": points[4] + segment_start_idx,
            f"{self.name}_f": points[5] + segment_start_idx,
            f"{self.name}_z": points[6] + segment_start_idx,
            f"{self.name}_q": points[7] + segment_start_idx,
            f"{self.name}_g": points[8] + segment_start_idx
        }

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        if not fiducials or f'{self.name}_a' not in fiducials:
            return False

        # Define QC window: Peak - 300 to Peak + 700
        peak_loc = fiducials[f'{self.name}_a']
        qc_start = max(0, peak_loc - 300)
        qc_end = min(len(preprocessed), peak_loc + 700)
        qc_window = preprocessed[qc_start:qc_end]
        
        if len(qc_window) == 0:
            return False

        # Peak relative to window start (is 300 unless we clipped at start)
        relative_peak_loc = 300 if peak_loc >= 300 else peak_loc

        try:
            qc_points = extract_fiducial_points(qc_window, c_peak=relative_peak_loc)
            point_amps = [qc_window[x] if x < len(qc_window) else 0 for x in qc_points]
            
            tests = [
                np.argmax(point_amps) in [0, 2],
                np.argmin(point_amps) == 1,
                (np.max(qc_window[:qc_points[2]]) - np.min(qc_window[:qc_points[2]]) * .5) > 
                (np.max(qc_window[qc_points[2]:]) - np.min(qc_window[qc_points[2]:])),
                qc_points[2] - qc_points[1] < 150,
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