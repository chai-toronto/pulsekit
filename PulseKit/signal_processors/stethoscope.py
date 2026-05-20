from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
from scipy.signal import find_peaks, hilbert
import numpy as np

class Stethoscope(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 40, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 240, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        pcg_segment = preprocessed[window_start + 10 : window_start + 150]
        
        if len(pcg_segment) == 0:
             return {}

        pcg_height_req = np.mean(np.abs(pcg_segment)) + np.std(np.abs(pcg_segment))
        pcg_peaks, _ = find_peaks(pcg_segment, distance=30, height=pcg_height_req)
        
        if len(pcg_peaks) > 0:
            return {f"{self.name}_s1": window_start + 10 + pcg_peaks[0]}
        
        return {}

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        """
        Validates the cardiac cycle by ensuring exactly two distinct peaks (S1 and S2) 
        exist within the ECG R-R interval envelope.
        """
        segment = preprocessed[window_start:window_end]
        if len(segment) < 10: return False

        envelope = np.abs(hilbert(segment))
        
        # Find all peaks above noise floor, pick top two (80 ms distance)
        all_peaks, properties = find_peaks(envelope, distance=int(0.08 * self.sampling_rate), height=np.mean(envelope))        
        if len(all_peaks) < 2:
            return False
        heights = properties['peak_heights']
        sorted_indices = np.argsort(heights)[::-1]
        primary_h = heights[sorted_indices[0]]
        secondary_h = heights[sorted_indices[1]]
        
        # if more than two peaks, get the 3rd, if 3rd peak is almost as tall as S1, there's not truly two peaks
        if len(all_peaks) > 2:
            tertiary_h = heights[sorted_indices[2]]
            if tertiary_h > (secondary_h * 0.7):
                return False
        
        # Relative Saliency Check: S1 must still be at least 15% of the max peak to ensure its not noise
        if (secondary_h / primary_h) < 0.15:
            return False

        # s1 within 200ms of r-peak, s2 after 200
        top_two_peaks = sorted([all_peaks[sorted_indices[0]], all_peaks[sorted_indices[1]]])
        sound_split = int(0.2 * self.sampling_rate)
        if top_two_peaks[0] > sound_split or top_two_peaks[1] < int(0.2 * self.sampling_rate):
            return False

        return True
    
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