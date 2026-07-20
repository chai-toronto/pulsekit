import numpy as np
import os
from typing import Dict, Any, Optional

class PhysiologicalSignal:
    '''
    Base class for physiological signals.
    :param self: PhysiologicalSignal object
    :param name: str, name of the signal
    :param sampling_rate: int, sampling rate in Hz (deafults to 1000)
    :param peaks: list, list of detected peaks in the signal
    :param data: np.ndarray, raw signal data
    :param preprocessed_data: np.ndarray, preprocessed signal data
    :param session_dir: str, directory path for the session data files
    :param is_loaded: bool, flag indicating if data is loaded in memory
    :param _custom_quality_check: Optional callable for custom quality check
    :param _custom_fiducial_extractor: Optional callable for custom fiducial extraction
    '''
    def __init__(self, name, sampling_rate=1000, session_dir=None):
        self.name = name
        self.sampling_rate = sampling_rate
        self.peaks = []
        self._data = None
        self.preprocessed_data = None
        self.session_dir = session_dir
        self.is_loaded = False
        self._custom_quality_check = None
        self._custom_fiducial_extractor = None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        '''
        Lazy loading of data. Loads data from file if not already loaded.
        
        :param self: PhysiologicalSignal object
        :return: np.ndarray, raw signal data
        '''
        if not self.is_loaded:
            self.load_data()    
        return self._data

    @data.setter
    def data(self, value) -> None:
        '''
        Sets the raw signal data and updates the is_loaded flag.
        Allows editing the data directly.
        
        :param self: PhysiologicalSignal object
        :param value: np.ndarray, new raw signal data
        '''
        self._data = value
        self.is_loaded = (value is not None)

    def load_data(self) -> None:
        '''
        Loads raw signal data from a file into memory.
        Default loader for numpy binary files. 
        Subclasses (like Audio) can override this if they use .wav or .csv
        
        :param self: PhysiologicalSignal object
        '''
        load_dir = os.path.join(self.session_dir, f"{self.name}.npy")
        if os.path.exists(load_dir):
            self._data = np.load(load_dir)
            self.is_loaded = True
        else:
            print(f"Warning: File not found at {load_dir}")

    def unload_data(self) -> None:
        '''
        Unloads raw signal data from memory to save space.
        
        :param self: PhysiologicalSignal object
        '''
        self.data = None
        self.is_loaded = False

    def preprocess(self) -> None:
        '''
        Placeholder method for preprocessing the signal data.
        
        :param self: PhysiologicalSignal object
        '''
        return self.data

    def get_sampling_rate(self) -> int:
        '''
        Retrieves the sampling rate of the signal.
        
        :param self: PhysiologicalSignal object
        '''
        return self.sampling_rate

    def get_peaks(self) -> list:
        '''
        Retrieves the detected peaks of the signal.
        
        :param self: PhysiologicalSignal object
        '''
        return self.peaks
    
    def set_peaks(self, peaks) -> None:
        '''
        Sets the detected peaks of the signal.
        
        :param self: PhysiologicalSignal object
        :param peaks: list, list of detected peaks
        '''
        self.peaks = peaks
    
    def get_activity_mask(self) -> Optional[np.ndarray]:
        '''
        Placeholder method for getting activity mask.
        
        :param self: PhysiologicalSignal object
        '''

        pass

    def set_activity_mask(self, mask) -> None:
        '''
        Placeholder method for setting activity mask.
        
        :param self: PhysiologicalSignal object
        '''

        pass

    def get_quality_mask(self) -> Optional[np.ndarray]:
        '''
        Placeholder method for getting quality mask.
        
        :param self: PhysiologicalSignal object
        '''

        pass
    
    def extract_fiducials(self, preprocessed, window_start: int, window_end: int, windowing_type: str) -> Optional[Dict[str, Any]]:
        '''
        Routes to specialized fiducial extraction methods based on windowing type.
        Methods like extract_cardiac_cycle_fiducials should be implemented in subclasses.
        If a custom extractor is provided, it will be used instead.
        
        :param self: PhysiologicalSignal object
        :param preprocessed: preprocessed signal data
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param windowing_type:  str, type of windowing used
        :return: Optional[Dict[str, Any]], extracted fiducials
        '''
        if self._custom_fiducial_extractor:
            return self._custom_fiducial_extractor(self, preprocessed, window_start, window_end, windowing_type)
        
        if windowing_type == "cardiac_cycles":
            return self.extract_cardiac_cycle_fiducials(preprocessed, window_start, window_end)
        elif windowing_type == "fixed_windows":
            return self.extract_fixed_window_fiducials(preprocessed, window_start, window_end)
        else:
            return None

    def has_removed_stretch(self, window_start: int, window_end: int, min_run_samples: Optional[int] = None) -> bool:
        '''
        Checks if there is a stretch of zeros in the raw signal data within the specified window, which may indicate a removed or corrupted segment.
        :param self: PhysiologicalSignal object
        :param window_start: int, index of window start
        :param window_end: int, index of window end
        :param min_run_samples: Optional[int], minimum number of consecutive zeros to consider as a removed stretch (defaults to 10 samples or 1% of sampling rate)
        :return: bool, True if a removed stretch is detected, False otherwise
        '''
        raw = self.data
        if raw is None:
            return False

        start = max(0, int(window_start))
        end = min(len(raw), int(window_end))
        if end <= start:
            return False

        segment = raw[start:end]
        if segment.size == 0:
            return False

        if min_run_samples is None:
            min_run_samples = max(10, int(0.01 * self.sampling_rate))

        zero_mask = np.asarray(segment == 0)
        if not np.any(zero_mask):
            return False

        padded = np.concatenate(([0], zero_mask.view(np.int8), [0]))
        changes = np.diff(padded)
        run_starts = np.flatnonzero(changes == 1)
        run_ends = np.flatnonzero(changes == -1)
        if run_starts.size == 0:
            return False

        run_lengths = run_ends - run_starts
        return bool(np.any(run_lengths >= min_run_samples))


    def check_quality(self, preprocessed, window_start: int, window_end: int, fiducials: dict, windowing_type: str) -> bool:
        '''
        Routes to specialized quality check methods based on windowing type.
        Methods like check_quality_cardiac_cycle should be implemented in subclasses.
        If a custom quality checker is provided, it will be used instead.
        
        :param self: PhysiologicalSignal object
        :param preprocessed: preprocessed signal data
        :param window_start: index of window start
        :type window_start: int
        :param window_end: index of window end
        :type window_end: int
        :param fiducials: fiducials extracted from the signal
        :type fiducials: dict
        :param windowing_type: type of windowing used
        :type windowing_type: str
        :return: returns whether the signal segment passes quality check
        :rtype: bool
        '''
        if self.has_removed_stretch(window_start, window_end):
            return False

        if self._custom_quality_check:
            return self._custom_quality_check(self, preprocessed, window_start, window_end, fiducials, windowing_type)
        
        if windowing_type == "cardiac_cycles":
            return self.check_quality_cardiac_cycle(preprocessed, window_start, window_end, fiducials)
        elif windowing_type == "fixed_windows":
            return self.check_quality_fixed_window(preprocessed, window_start, window_end, fiducials)
        else:
            raise NotImplementedError(f"Quality check for windowing type '{windowing_type}' is not implemented.")