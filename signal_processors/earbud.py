from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
import numpy as np
from scipy.stats import kurtosis
from typing import Dict, Any
from .base import PhysiologicalSignal

class Earbud(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        # Logic from: preprocess_earbud
        self.data = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        self.data = apply_butterworth_filter(self.data, 15, "low", fs=self.sampling_rate)

    def analyze_beat(self, beat_start: int, beat_end: int) -> Dict[str, Any]:
        # 1. Extraction Logic from: extract_earbud
        # ---------------------------------------------------------
        theoretical_min = int(0.1 * 0.3 * self.sampling_rate) - 10 
        theoretical_max = 500 
        
        win_start = beat_start + theoretical_min
        win_end = min(beat_start + theoretical_max, len(self.data))
        arrival_window = self.data[win_start : win_end]
        
        if len(arrival_window) == 0:
            return {"quality_check_passed": False}

        a_peak_global = np.argmax(arrival_window) + win_start
        
        segment_start_idx = max(0, beat_start - 200)
        segment_end_idx = min(len(self.data), beat_end + 200)
        segment = self.data[segment_start_idx : segment_end_idx]
        
        anchor_rel = a_peak_global - segment_start_idx
        points = extract_fiducial_points(segment, c_peak=anchor_rel)
        
        results = {
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

        # 2. Quality Check Logic from: check_earbud & process_signal
        # ---------------------------------------------------------
        # Define QC window: Peak - 300 to Peak + 700
        # The 'a' point in results is equivalent to 'peak' in the old logic
        peak_loc = results['a']
        qc_start = max(0, peak_loc - 300)
        qc_end = min(len(self.data), peak_loc + 700)
        qc_window = self.data[qc_start:qc_end]
        
        # QC requires peak relative to window start (which is 300 unless we clipped at start)
        relative_peak_loc = 300 if peak_loc >= 300 else peak_loc

        try:
            # Re-extract on QC window
            qc_points = extract_fiducial_points(qc_window, c_peak=relative_peak_loc)
            point_amps = [qc_window[x] if x < len(qc_window) else 0 for x in qc_points]
            
            tests = [
                np.argmax(point_amps) in [0, 2],
                np.argmin(point_amps) == 1,
                (np.max(qc_window[:qc_points[2]]) - np.min(qc_window[:qc_points[2]]) * .5) > 
                (np.max(qc_window[qc_points[2]:]) - np.min(qc_window[qc_points[2]:])),
                qc_points[2] - qc_points[1] < 150,
            ]
            passed = all(tests)
        except Exception:
            passed = False
            
        results["quality_check_passed"] = passed
        return results