from .base_signal import PhysiologicalSignal
from .signal_processing_utils import cleanup_bp_signal_rm 
import numpy as np
import scipy.ndimage as ndimage
import os

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
        self.artifact_integral = None

    def preprocess(self) -> np.ndarray:
        """
        Preprocess the BP signal by loading the external NIBP mask and combining it 
        with standard signal cleaning (NaNs).
        """
        raw_data = self.data
    
        
        # 1. Load the pre-calculated NIBP mask from the session folder
        external_mask = self._load_external_nibp_mask()
        
        # 2. Run standard cleanup (removes sharp spikes/impossible values)
        _, _, cleaned_data = cleanup_bp_signal_rm(raw_data)
        
        # 3. Identify NaNs from the cleanup process
        cleaned_mask = np.isnan(cleaned_data)
        
        # 4. Combine Masks
        # external_mask is True for GOOD data, so we invert it for the ARTIFACT mask
        # self.artifact_mask will be True where data should be REJECTED
        self.artifact_mask = (~external_mask) | cleaned_mask 

        # 5. Generate integral for fast window quality lookups
        self.artifact_integral = np.cumsum(self.artifact_mask.astype(int))
        self.artifact_integral = np.insert(self.artifact_integral, 0, 0)

        return raw_data

    def _load_external_nibp_mask(self) -> np.ndarray:
        """
        Loads the consolidated NIBP mask (CNAP_Annotations.npy) from the session directory.
        If the file is missing, returns an all-True (all good) mask of the same length.
        """
        mask_path = os.path.join(self.session_dir, "CNAP_Annotations.npy")
        
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            mask = np.asarray(mask, dtype=bool)
            return mask
        
        # Fallback: If no mask found, assume all data is good to prevent crashes
        return np.ones(len(self.data), dtype=bool)

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

        artifacts_in_window = self.artifact_integral[window_end] - self.artifact_integral[window_start]
        if artifacts_in_window > 0:
            return False
        
        val = fiducials.get(f"{self.name}_midpoint", 0)
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