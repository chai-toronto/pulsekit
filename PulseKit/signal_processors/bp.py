from .base_signal import PhysiologicalSignal
from .signal_processing_utils import cleanup_bp_signal_rm 
import numpy as np
import scipy.ndimage as ndimage

class BP(PhysiologicalSignal):
    '''
    Blood Pressure Signal Processor.
    Inherits from PhysiologicalSignal.
    For use with Diastolic_BP, Systolic_BP, or NIBP signals.

    :param self: BP object
    :param name: str, name of the signal
    :param sampling_rate: int, sampling rate in Hz
    :param session_dir: str, directory path for the session data files
    :param artifact_mask: np.ndarray, boolean mask indicating artifact regions, this is calculated uniquely for this signal type
    '''
    def __init__(self, name, sampling_rate=1000, session_dir=None):
        super().__init__(name, sampling_rate, session_dir)
        self.artifact_mask = None 

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the BP signal by cleaning and generating calibration artifact mask.
        Sets self.artifact_mask attribute

        :param self: BP object
        :return: np.ndarray, raw BP signal data
        """
        raw_data = self.data
        calibration_mask = self._detect_calibration_artifacts(raw_data)
        _, _, cleaned_data = cleanup_bp_signal_rm(raw_data)
        cleaned_mask = np.isnan(cleaned_data)
        self.artifact_mask = calibration_mask | cleaned_mask

        return raw_data

    def _detect_calibration_artifacts(self, data, min_run_seconds=5, buffer_seconds=20) -> np.ndarray:
        """
        Identifies calibration events (>5 second runs of exactly 50 or 150) and buffers them by 20 seconds
        Private method used in preprocess()

        :param self: BP object
        :param data: np.ndarray, raw BP signal data
        :param min_run_seconds: int, minimum duration in seconds to consider as calibration event
        :param buffer_seconds: int, duration in seconds to buffer before and after calibration event
        :return: np.ndarray, boolean mask indicating calibration artifact regions
        """

        is_50 = (data >= 49) & (data <= 51)
        is_150 = (data >= 149) & (data <= 151)
        raw_mask = is_50 | is_150
        min_run_samples = int(min_run_seconds * self.sampling_rate)
        long_run_mask = ndimage.binary_opening(raw_mask, structure=np.ones(min_run_samples))
        buffer_samples = int(buffer_seconds * self.sampling_rate)
        dilated_mask = ndimage.binary_dilation(long_run_mask, structure=np.ones(buffer_samples))
        
        return dilated_mask

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int) -> dict:
        """
        Extracts Mean, Midpoint, and Median from the BP signal within the given window.
        Only relevant to cardiac cycle windowing.

        :param self: BP object
        :param preprocessed: np.ndarray, preprocessed BP signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :return: dict, extracted fiducials
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

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict, min_value:int=20, max_value:int=250) -> bool:
        """
        Checks if the current window overlaps with any known artifacts.
        Verifies normal physiological range for BP values.

        :param self: BP object
        :param preprocessed: np.ndarray, preprocessed BP signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param fiducials: dict, extracted fiducials
        :param min_value: int, minimum acceptable BP value
        :param max_value: int, maximum acceptable BP value
        :return: bool, True if the window is of good quality, False otherwise
        """
        if window_end > len(self.artifact_mask):
            return False

        window_mask = self.artifact_mask[window_start:window_end]
        if np.any(window_mask):
            return False 
        
        val = fiducials.get("midpoint", 0)
        if val < 20 or val > 250:
            return False

        return True
        
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        '''
        Placeholder for fixed window quality check.
        Same logic as cardiac cycle quality check for now.

        :param self: BP object
        :param preprocessed: np.ndarray, preprocessed BP signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param fiducials: dict, extracted fiducials
        :return: bool, True if the window is of good quality, False otherwise
        '''
        return self.check_quality_cardiac_cycle(preprocessed, window_start, window_end, fiducials)