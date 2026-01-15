from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import kurtosis
from typing import Dict, Any
from .base import PhysiologicalSignal

class PPG(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        # Logic from: preprocess_ppg
        self.data = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        self.data = apply_butterworth_filter(self.data, 15, "low", fs=self.sampling_rate)

    def analyze_beat(self, beat_start: int, beat_end: int) -> Dict[str, Any]:
        # 1. Extraction Logic from: extract_ppg
        # ---------------------------------------------------------
        search_window_end = min(len(self.data), beat_end + 500)
        clip = self.data[beat_start:search_window_end]
        
        # Flip and find feet
        flipped_clip = -clip
        flipped_mean = np.mean(flipped_clip)
        feet, _ = find_peaks(flipped_clip, height=flipped_mean, distance=200)
        
        if len(feet) < 2:
            return {"quality_check_passed": False}

        foot1 = feet[0]
        foot2 = feet[1]
        
        # Find systolic peak
        head_idx = np.argmax(self.data[beat_start + foot1 : beat_start + foot2])
        
        # Derivative based extraction
        segment_start_idx = max(0, beat_start + foot1 - 200)
        segment_end_idx = beat_start + foot2
        segment = self.data[segment_start_idx : segment_end_idx]

        ppg_derivative = np.diff(segment)
        ppg_second_deriv = np.diff(ppg_derivative)
        
        # Find 'a' peak on second derivative
        # TODO: Add safety check if ppg_second_deriv is empty
        if len(ppg_second_deriv) == 0:
             return {"quality_check_passed": False}
             
        a_peak_rel = np.argmax(ppg_second_deriv)
        points = extract_fiducial_points(ppg_second_deriv, c_peak=a_peak_rel)

        results = {
            "peak": beat_start + foot1 + head_idx,
            "foot": beat_start + foot1,
            "foot_next": beat_start + foot2,
            # Map relative points back to global indices
            "a": points[0] + segment_start_idx,
            "b": points[1] + segment_start_idx,
            "c": points[2] + segment_start_idx,
            "d": points[3] + segment_start_idx,
            "e": points[4] + segment_start_idx,
            "f": points[5] + segment_start_idx,
            "z": points[6] + segment_start_idx,
            "q": points[7] + segment_start_idx,
            "g": points[8] + segment_start_idx
        }

        # 2. Quality Check Logic from: check_ppg & process_signal
        # ---------------------------------------------------------
        # Define QC window (foot to foot_next)
        qc_start = results['foot']
        qc_end = results['foot_next']
        qc_window = self.data[qc_start:qc_end]

        # Specific PPG QC math
        # Note: Recalculating derivatives here as per original check_ppg logic
        w_diff = np.diff(qc_window, prepend=qc_window[0])
        w_diff2 = np.diff(w_diff, prepend=w_diff[0])
        w_filtered = apply_butterworth_filter(w_diff2, 25, "low", fs=self.sampling_rate)

        try:
            # Re-extract points on this specific QC window to verify morphology
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
            passed = all(tests)
        except Exception:
            passed = False

        results["quality_check_passed"] = passed
        return results