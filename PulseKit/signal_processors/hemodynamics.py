from .base_signal import PhysiologicalSignal
import numpy as np

class Hemodynamics(PhysiologicalSignal):
    '''
    Catch-all signal processor for interpreted hemodynamic signals (Stroke Volume, Cardiac Output, SVR, and Heart Rate)
    '''

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int) -> dict:
        '''
        Extracts mean, median, and midpoint values from the hemodynamics signal within the cardiac window.
        
        :param self: Hemodynamics object
        :param preprocessed: np.ndarray, preprocessed hemodynamics signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :return: dict, extracted fiducials
        '''
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
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int) -> dict:
        """
        Extracts Mean, Midpoint, and Median from the BP signal within the given window.

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

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> bool:
        '''
        Checks the quality of the hemodynamics signal within the cardiac window.
        Checks to see if there are any zeros in the window as a basic quality check.
        May require additional physiological range checks depending on the specific hemodynamic parameter.
        
        :param self: Hemodynamics object
        :param preprocessed: np.ndarray, preprocessed BP signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param fiducials: dict, extracted fiducials
        :return: bool, True if the window is of good quality, False otherwise
        '''
        window = preprocessed[window_start:window_end]

        return not np.any(window == 0)
    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> bool:
        '''
        Placeholder for fixed window quality check.
        Same logic as cardiac cycle quality check for now.

        :param self: Hemodynamics object
        :param preprocessed: np.ndarray, preprocessed BP signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param fiducials: dict, extracted fiducials
        :return: bool, True if the window is of good quality, False otherwise
        '''
        return self.check_quality_cardiac_cycle(preprocessed, window_start, window_end, fiducials)