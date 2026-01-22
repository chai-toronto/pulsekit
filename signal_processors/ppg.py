from .base import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
from scipy.signal import find_peaks
import numpy as np

class PPG(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 15, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        # Safety for edge of signal
        search_window_end = min(len(preprocessed), window_end + 500)
        clip = preprocessed[window_start:search_window_end]
        
        # 1. Find Feet (Flip signal)
        flipped_clip = -clip
        flipped_mean = np.mean(flipped_clip)
        feet, _ = find_peaks(flipped_clip, height=flipped_mean, distance=200)
        
        if len(feet) < 2:
            return {} # Failed to find cycle boundaries

        foot1 = feet[0]
        foot2 = feet[1]
        
        # 2. Find Systolic Peak
        head_idx = np.argmax(preprocessed[window_start + foot1 : window_start + foot2])
        
        # 3. Derivative Analysis for Morphological Points (a, b, c...)
        segment_start_idx = max(0, window_start + foot1 - 200)
        segment_end_idx = window_start + foot2
        segment = preprocessed[segment_start_idx : segment_end_idx]

        ppg_derivative = np.diff(segment)
        ppg_second_deriv = np.diff(ppg_derivative)
        
        if len(ppg_second_deriv) == 0:
             return {}

        a_peak_rel = np.argmax(ppg_second_deriv)
        points = extract_fiducial_points(ppg_second_deriv, c_peak=a_peak_rel)

        # Map relative points back to global indices
        return {
            f"{self.name}_peak": window_start + foot1 + head_idx,
            f"{self.name}_foot": window_start + foot1,
            f"{self.name}_foot_next": window_start + foot2,
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
        if not fiducials or f'{self.name}_foot' not in fiducials or f'{self.name}_foot_next' not in fiducials:
            return False

        # Define QC window (foot to foot_next)
        qc_start = fiducials[f'{self.name}_foot']
        qc_end = fiducials[f'{self.name}_foot_next']
        qc_window = preprocessed[qc_start:qc_end]

        if len(qc_window) == 0: 
            print("QC window length 0")
            return False

        try:
            # Re-calculate derivatives specific to QC logic
            w_diff = np.diff(qc_window, prepend=qc_window[0])
            w_diff2 = np.diff(w_diff, prepend=w_diff[0])
            w_filtered = apply_butterworth_filter(w_diff2, 25, "low", fs=self.sampling_rate)

            # Verify morphology
            a_peak_qc = np.argmax(w_filtered[:int(w_filtered.shape[0]/2)])
            qc_points = extract_fiducial_points(w_filtered, c_peak=a_peak_qc)
            point_amps = [w_filtered[x] if x < len(w_filtered) else 0 for x in qc_points]
            
            tests = [
                np.argmax(point_amps) in [0, 2],
                np.argmin(point_amps) == 1,
                (np.max(w_filtered[:qc_points[2]]) - np.min(w_filtered[:qc_points[2]]) * .5) > 
                (np.max(w_filtered[qc_points[2]:]) - np.min(w_filtered[qc_points[2]:])),
                qc_points[2] - qc_points[1] < 150,
            ]
            return all(tests)
        except Exception as e:
            return False
        
    def check_ppg(self, window, **kwargs):
        window = np.diff(window, prepend=window[0])
        window = np.diff(window, prepend=window[0])
        window = apply_butterworth_filter(window, 25, "low", 1000)

        a_peak = np.argmax(window[:int(window.shape[0]/2)])
        fiducial_points = extract_fiducial_points(window, anchor=a_peak)
        fiducial_point_amplitudes = [window[x] if x < len(window) else 0 for x in fiducial_points]
        try:
            tests = [
                np.argmax(fiducial_point_amplitudes) in [0,2],
                np.argmin(fiducial_point_amplitudes) == 1,
                np.max(window[:fiducial_points[2]])-np.min(window[:fiducial_points[2]])*.5 > np.max(window[fiducial_points[2]:])-np.min(window[fiducial_points[2]:]),
                fiducial_points[2]-fiducial_points[1] < 150,
            ]
        except Exception as e:
            tests = [False, False, False, False]

        return all(tests)
        
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