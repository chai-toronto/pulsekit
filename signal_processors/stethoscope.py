from .signal_processing_utils import apply_butterworth_filter
from scipy.signal import find_peaks
import numpy as np
from scipy.stats import kurtosis
from typing import Dict, Any
from .base import PhysiologicalSignal

class Stethoscope(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        # Logic from: preprocess_stethoscope
        self.data = apply_butterworth_filter(self.data, 40, "high", fs=self.sampling_rate)
        self.data = apply_butterworth_filter(self.data, 100, "low", fs=self.sampling_rate)

    def analyze_beat(self, beat_start: int, beat_end: int) -> Dict[str, Any]:
        # Extraction Logic from: extract_stethoscope
        pcg_segment = self.data[beat_start + 10 : beat_start + 150]
        
        # TODO: Handle edge case where segment is empty or too short
        if len(pcg_segment) == 0:
             return {"quality_check_passed": False}

        pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
        pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
        
        s1 = None
        if len(pcg_peaks) > 0:
            s1 = beat_start + 10 + pcg_peaks[0]
            
        fiducials = {"s1": s1} if s1 is not None else {}

        # QC Logic from: check_stethoscope
        window = self.data[beat_start:beat_end]
        passed = (np.std(window) > 0.001)

        return {**fiducials, "quality_check_passed": passed}