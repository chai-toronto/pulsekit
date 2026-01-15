from .base import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
from scipy.signal import find_peaks
import numpy as np

class Stethoscope(PhysiologicalSignal):
    def preprocess(self):
        super().preprocess()
        processed = apply_butterworth_filter(self.data, 40, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 100, "low", fs=self.sampling_rate)

    def extract_fiducials(self, preprocessed, beat_start: int, beat_end: int):
        pcg_segment = preprocessed[beat_start + 10 : beat_start + 150]
        
        if len(pcg_segment) == 0:
             return {}

        pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
        pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
        
        if len(pcg_peaks) > 0:
            return {f"{self.name}_s1": beat_start + 10 + pcg_peaks[0]}
        
        return {}

    def check_quality(self, preprocessed, beat_start: int, beat_end: int, fiducials: dict):
        window = preprocessed[beat_start:beat_end]
        # Basic check: is there signal variation?
        return (np.std(window) > 0.001)