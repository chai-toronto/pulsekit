from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter
import numpy as np
from scipy.signal import find_peaks, hilbert, welch
from scipy.stats import pearsonr

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
        
        pep_ms = (ao_local / self.sampling_rate) * 1000
        if pep_ms < 60 or pep_ms > 150:
            return False
            
        ivct_ms = ((ao_local - im_local) / self.sampling_rate) * 1000
        if ivct_ms < 20 or ivct_ms > 90:
            return False

        return True

    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> float:
        '''
        Calculates a Signal Quality Index (SQI) for a fixed multi-beat SCG window
        using Beat-by-Beat Adaptive Template Matching. Returns a float between 0.0 and 1.0.
        Anchors directly on the SCG 'AO peak' within autonomous Hilbert energy bursts.
        '''
        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        if len(sig) < fs:
            return 0.0
        if not np.isfinite(sig).all():
            return 0.0

        # Use the Hilbert envelope to locate the high-frequency systolic energy bursts
        env = np.abs(hilbert(sig))
        height_req = np.mean(env)
        min_peak_dist = int(0.4 * fs) # Assumes max HR ~150 BPM
        
        complex_centers, _ = find_peaks(env, height=height_req, distance=min_peak_dist)
        
        true_ao_peaks = []
        for center in complex_centers:
            # Look locally around the energy burst to find the true physiological peaks
            search_start = max(0, center - int(0.15 * fs))
            search_end = min(len(sig), center + int(0.15 * fs))
            local_region = sig[search_start:search_end]
            
            if len(local_region) == 0: 
                continue
                
            pos_peaks, _ = find_peaks(local_region, distance=10, prominence=0.5 * np.mean(np.abs(local_region)))
            
            if len(pos_peaks) > 0:
                ao_idx = pos_peaks[np.argmax(local_region[pos_peaks])]
            else:
                ao_idx = np.argmax(local_region)
                
            true_ao_peaks.append(search_start + ao_idx)
            
        # Require at least 3 valid beats to form a reliable consensus template
        if len(true_ao_peaks) < 3:
            return 0.0
            
        pre_samples = int(0.20 * fs)
        post_samples = int(0.40 * fs)
        
        beat_matrix = []
        for peak in true_ao_peaks:
            if peak - pre_samples < 0 or peak + post_samples >= len(sig):
                continue
            beat_segment = sig[peak - pre_samples : peak + post_samples]
            beat_matrix.append(beat_segment)
            
        beat_matrix = np.array(beat_matrix)
        
        if len(beat_matrix) < 3:
            return 0.0
            
        local_template = np.median(beat_matrix, axis=0)
        
        correlations = []
        for beat in beat_matrix:
            r, _ = pearsonr(beat, local_template)
            
            if np.isnan(r):
                r = 0.0
            # Clip negative correlations to 0
            correlations.append(max(0.0, r))
            
        window_sqi = float(np.mean(correlations))
        
        return window_sqi
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        """
        Extracts heart rate from accelerometer within the given window.
        
        param self: accelerometer object
        param preprocessed: np.ndarray, preprocessed accelerometer signal data
        param window_start: int, index of window start
        param window_end: int, index of window end
        """
        from scipy.signal import welch, find_peaks, hilbert
        import numpy as np

        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        if len(sig) == 0:
            return {f"{self.name}_HR": np.nan}

        env = np.abs(hilbert(sig))
        
        # 150ms moving avg to fuse the SCG complex
        win_len = int(0.15 * fs)
        smoothed_env = np.convolve(env, np.ones(win_len)/win_len, mode='same')
        
        # 1.5s moving avg for baseline shifts
        roll_len = int(fs * 1.5)
        local_mean = np.convolve(smoothed_env, np.ones(roll_len)/roll_len, mode='same')
        
        # Threshold: 20% above local mean
        dynamic_threshold = local_mean * 1.2 + 1e-4
        min_peak_dist = int(0.4 * fs)
        
        complex_centers, _ = find_peaks(smoothed_env, height=dynamic_threshold, distance=min_peak_dist)
        
        # Extract true AO peaks
        true_ao_peaks = []
        search_radius = int(0.15 * fs)
        
        for center in complex_centers:
            search_start = max(0, center - search_radius)
            search_end = min(len(sig), center + search_radius)
            local_region = sig[search_start:search_end]
            
            if len(local_region) == 0: 
                continue
                
            local_ptp = np.ptp(local_region)
            if local_ptp == 0: 
                continue
                
            pos_peaks, _ = find_peaks(local_region, distance=int(0.05 * fs), prominence=0.2 * local_ptp)
            
            if len(pos_peaks) > 0:
                ao_idx = pos_peaks[np.argmax(local_region[pos_peaks])]
                true_ao_peaks.append(search_start + ao_idx)
            
        # Calculate HR from IBI
        if len(true_ao_peaks) >= 4:
            ibis_in_samples = np.diff(true_ao_peaks)
            ibis_in_seconds = ibis_in_samples / fs
            
            if len(ibis_in_seconds) >= 3:
                # Rhythm regularity check (CV < 15%)
                cv = np.std(ibis_in_seconds) / np.mean(ibis_in_seconds)
                
                if cv < 0.4: 
                    median_ibi = np.median(ibis_in_seconds)
                    hr_bpm_ibi = float(60.0 / median_ibi)
                else:
                    hr_bpm_ibi = np.nan 
            else:
                hr_bpm_ibi = np.nan 
        else:
            hr_bpm_ibi = np.nan

        return {
            f"{self.name}_HR": hr_bpm_ibi
        }