from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
import numpy as np
from scipy.signal import find_peaks, hilbert

class Accelerometer(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.8, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 25, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        scg_seg = preprocessed[window_start:window_end]
        if len(scg_seg) == 0:
            return {}
        
        im_search_start = int(0.015 * self.sampling_rate)
        im_search_end = int(0.15 * self.sampling_rate)
        im_search_region = scg_seg[im_search_start:im_search_end]
        neg_peaks, _ = find_peaks(-im_search_region, distance=10, prominence=0.2 * np.mean(np.abs(im_search_region)))
        im_idx = neg_peaks[0] if len(neg_peaks) > 0 else np.argmin(im_search_region)
        im_idx += im_search_start
        
        ao_search_len = int(0.1 * self.sampling_rate)  # Search for AO in next 100ms
        ao_search_region = scg_seg[im_idx : im_idx + ao_search_len]
        pos_peaks, _ = find_peaks(ao_search_region, distance=10, prominence=0.5 * np.mean(np.abs(ao_search_region)))
        ao_idx = (pos_peaks[0] + im_idx) if len(pos_peaks) > 0 else (np.argmax(ao_search_region) + im_idx)
        
        return {
            f"{self.name}_IM": im_idx + window_start,
            f"{self.name}_AO": ao_idx + window_start
            }

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        im_local = fiducials.get(f"{self.name}_IM", 0) - window_start
        ao_local = fiducials.get(f"{self.name}_AO", 0) - window_start
            
        # 3. PEP Sanity Bounds (Pre-Ejection Period)
        # Your R-AO mean is ~98 ms (Std ~11.6 ms). 
        # A range of 60 to 150 ms covers +/- 3 standard deviations while aligning w literature
        pep_ms = (ao_local / self.sampling_rate) * 1000
        if pep_ms < 60 or pep_ms > 150:
            return False
            
        # 4. IVCT Sanity Bounds (Isovolumetric Contraction Time)
        # Your IM-AO mean is ~62.5 ms (Std ~6 ms). 
        # Extending the ceiling to 90 ms easily captures +/- 4 standard deviations, but filters outliers
        ivct_ms = ((ao_local - im_local) / self.sampling_rate) * 1000
        if ivct_ms < 20 or ivct_ms > 90:
            return False

        # 5. Amplitude Check (Noise Floor)
        # Ensure the AO peak isn't just a tiny ripple in a noisy signal
        #window = preprocessed[window_start:window_end]
        #if len(window) > 0:
        #    ao_amp = np.abs(preprocessed[int(fiducials[f"{self.name}_AO"])])
        #    mean_abs_amp = np.mean(np.abs(window))
        #    if ao_amp < 1.5 * mean_abs_amp: # AO should be a prominent local event
        #        return False

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