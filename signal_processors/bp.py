from .base import PhysiologicalSignal
from .signal_processing_utils import cleanup_bp_signal_rm 
import numpy as np
import scipy.ndimage as ndimage

class BP(PhysiologicalSignal):
    def __init__(self, name, sampling_rate, session_dir=None):
        super().__init__(name, sampling_rate, session_dir)
        self.artifact_mask = None 

    def preprocess(self):
        """
        Preprocess the BP signal by cleaning and generating calibration artifact mask.
        """
        raw_data = self.data
        calibration_mask = self._detect_calibration_artifacts(raw_data)
        _, _, cleaned_data = cleanup_bp_signal_rm(raw_data)
        cleaned_mask = np.isnan(cleaned_data)
        self.artifact_mask = calibration_mask | cleaned_mask

        return raw_data

    def _detect_calibration_artifacts(self, data):
        """
        Identifies calibration events (>30 second runs of ~50 or ~150) and buffers them.
        """
        is_50 = (data >= 49) & (data <= 51)
        is_150 = (data >= 149) & (data <= 151)
        
        raw_mask = is_50 | is_150
        buffer_seconds = 30
        buffer_samples = int(buffer_seconds * self.sampling_rate)
        dilated_mask = ndimage.binary_dilation(raw_mask, structure=np.ones(buffer_samples))
        print(f"Detected {np.sum(raw_mask)} calibration samples, dilated to {np.sum(dilated_mask)} samples out of {len(data)} total.")
        
        return dilated_mask

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        """
        Extracts Mean, Midpoint, and Median.
        """
        window_data = preprocessed[window_start:window_end]
        
        if len(window_data) == 0:
            return {"mean": np.nan, "midpoint": np.nan, "median": np.nan}
        
        val_mean = np.mean(window_data)
        val_median = np.median(window_data)
        mid_idx = len(window_data) // 2
        val_midpoint = window_data[mid_idx]

        return {
            f"{self.name}_mean": val_mean,
            f"{self.name}_median": val_median,
            f"{self.name}_midpoint": val_midpoint
        }

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        """
        Checks if the current window overlaps with any known artifacts.
        """
        if window_end > len(self.artifact_mask):
            return False

        # Check if ANY point in this window is marked as an artifact
        window_mask = self.artifact_mask[window_start:window_end]
        if np.any(window_mask):
            return False 
        
        # sanity check
        val = fiducials.get("midpoint", 0)
        if val < 20 or val > 250:
            return False

        return True
        
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        # Re-use the same logic as cardiac cycle
        return self.check_quality_cardiac_cycle(preprocessed, window_start, window_end, fiducials)
    
    def extract_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        # Re-use the same logic as cardiac cycle
        return self.extract_cardiac_cycle_fiducials(preprocessed, window_start, window_end)