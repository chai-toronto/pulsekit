from .base_signal import PhysiologicalSignal
from .signal_processing_utils import apply_butterworth_filter, extract_fiducial_points
from scipy.signal import find_peaks, welch
import numpy as np
from scipy.stats import pearsonr

class PPG(PhysiologicalSignal):
    def preprocess(self):
        processed = apply_butterworth_filter(self.data, 0.5, "high", fs=self.sampling_rate)
        return apply_butterworth_filter(processed, 15, "low", fs=self.sampling_rate)

    def extract_cardiac_cycle_fiducials(self, preprocessed, window_start: int, window_end: int):
        fs = self.sampling_rate
        
        # Buffer to catch the 'next' foot (Pulse transit time pushes the foot past the next ECG peak)
        # Using 0.5 seconds to safely capture the downstream beat
        search_window_end = min(len(preprocessed), window_end + int(fs * 0.5))
        clip = preprocessed[window_start:search_window_end]
        
        if len(clip) < int(fs * 0.3): 
            return {}
        
        # 1. Find Feet using First Derivative Minima Search
        ppg_diff = np.diff(clip)
        
        # Find steepest slopes (systolic upstrokes). 
        # Distance ensures we don't catch dicrotic notches.
        diff_peaks, _ = find_peaks(ppg_diff, height=np.max(ppg_diff)*0.2, distance=int(fs * 0.4))
        
        feet = []
        for dp in diff_peaks:
            foot_idx = dp
            search_limit = max(0, dp - int(fs * 0.3)) # Look back max 300ms
            
            # Trace backward until the derivative drops to zero or below
            while foot_idx > search_limit and ppg_diff[foot_idx] > 0:
                foot_idx -= 1
                
            feet.append(foot_idx)
            
        if len(feet) < 2:
            return {} # Failed to find cycle boundaries

        foot1 = feet[0]
        foot2 = feet[1]
        
        # 2. Find Systolic Peak
        # head_idx is relative to the start of foot1
        head_idx = np.argmax(preprocessed[window_start + foot1 : window_start + foot2])
        
        # 3. Derivative Analysis for Morphological Points (a, b, c...)
        segment_start_idx = max(0, window_start + foot1 - int(fs * 0.2)) 
        segment_end_idx = window_start + foot2
        segment = preprocessed[segment_start_idx : segment_end_idx]

        ppg_derivative = np.diff(segment)
        ppg_second_deriv = np.diff(ppg_derivative)
        
        if len(ppg_second_deriv) == 0:
             return {}

        a_peak_rel = np.argmax(ppg_second_deriv)
        points = extract_fiducial_points(ppg_second_deriv, c_peak=a_peak_rel)

        # Map relative points back to global indices
        return {
            f"{self.name}_peak": window_start + foot1 + head_idx,
            f"{self.name}_foot": window_start + foot1,
            f"{self.name}_foot_next": window_start + foot2,
            f"{self.name}_a": points[0] + segment_start_idx,
            f"{self.name}_b": points[1] + segment_start_idx,
            f"{self.name}_c": points[2] + segment_start_idx,
            f"{self.name}_d": points[3] + segment_start_idx,
            f"{self.name}_e": points[4] + segment_start_idx,
            f"{self.name}_f": points[5] + segment_start_idx,
            f"{self.name}_z": points[6] + segment_start_idx,
            f"{self.name}_q": points[7] + segment_start_idx,
            f"{self.name}_g": points[8] + segment_start_idx
        }

    def check_quality_cardiac_cycle(self, preprocessed, window_start: int, window_end: int, fiducials: dict):
        if not fiducials or f'{self.name}_foot' not in fiducials or f'{self.name}_foot_next' not in fiducials:
            return False

        # Define QC window (foot to foot_next)
        qc_start = fiducials[f'{self.name}_foot']
        qc_end = fiducials[f'{self.name}_foot_next']
        qc_window = preprocessed[qc_start:qc_end]

        if len(qc_window) == 0: 
            print("QC window length 0")
            return False

        try:
            # Re-calculate derivatives specific to QC logic
            w_diff = np.diff(qc_window, prepend=qc_window[0])
            w_diff2 = np.diff(w_diff, prepend=w_diff[0])
            w_filtered = apply_butterworth_filter(w_diff2, 25, "low", fs=self.sampling_rate)

            # Verify morphology
            a_peak_qc = np.argmax(w_filtered[:int(w_filtered.shape[0]/2)])
            qc_points = extract_fiducial_points(w_filtered, c_peak=a_peak_qc)
            point_amps = [w_filtered[x] if x < len(w_filtered) else 0 for x in qc_points]
            
            tests = [
                np.argmax(point_amps) in [0, 2],
                np.argmin(point_amps) == 1,
                (np.max(w_filtered[:qc_points[2]]) - np.min(w_filtered[:qc_points[2]]) * .5) > 
                (np.max(w_filtered[qc_points[2]:]) - np.min(w_filtered[qc_points[2]:])),
                qc_points[2] - qc_points[1] < 150,
            ]
            return all(tests)
        except Exception as e:
            return False
        
    
    def check_quality_fixed_window(self, preprocessed, window_start: int, window_end: int, fiducials: dict) -> float:
        '''
        Calculates a Signal Quality Index (SQI) for a fixed multi-beat window
        using Beat-by-Beat Adaptive Template Matching. Returns a float between 0.0 and 1.0.
        Anchors on the PPG systolic peak (extracted via foot-to-foot bounds).
        '''
        sig = preprocessed[window_start:window_end]
        fs = self.sampling_rate
        
        # Failsafe for extremely short windows
        if len(sig) < fs:
            return 0.0

        sig_diff = np.diff(sig)
        
        # Avoid crashing if the signal is a complete flatline
        max_diff = np.max(sig_diff)
        if max_diff == 0:
            return 0.0
            
        height_req = max_diff * 0.2
        min_peak_dist = int(0.4 * fs) # Assumes max HR ~150 BPM
        
        diff_peaks, _ = find_peaks(sig_diff, height=height_req, distance=min_peak_dist)
        
        # Find feet by tracing backward from steepest derivative
        feet = []
        for dp in diff_peaks:
            foot_idx = dp
            search_limit = max(0, dp - int(fs * 0.3)) # Look back max 300ms
            
            # Trace backward until the derivative drops to zero or below
            while foot_idx > search_limit and sig_diff[foot_idx] > 0:
                foot_idx -= 1
                
            feet.append(foot_idx)
            
        # Extract true systolic peaks strictly between consecutive feet
        true_peaks = []
        if len(feet) >= 2:
            for i in range(len(feet) - 1):
                foot1 = feet[i]
                foot2 = feet[i+1]
                
                # Failsafe to prevent empty slice errors
                if foot1 < foot2:
                    head_idx = np.argmax(sig[foot1 : foot2])
                    true_peaks.append(foot1 + head_idx)

        # Require at least 3 valid beats to form a reliable consensus template
        if len(true_peaks) < 3:
            return 0.0
            
        #  Beat Segmentation
        pre_samples = int(0.30 * fs)
        post_samples = int(0.50 * fs)
        
        beat_matrix = []
        for peak in true_peaks:
            # Discard beats that fall off the edge of the fixed window
            if peak - pre_samples < 0 or peak + post_samples >= len(sig):
                continue
            beat_segment = sig[peak - pre_samples : peak + post_samples]
            beat_matrix.append(beat_segment)
            
        beat_matrix = np.array(beat_matrix)
        
        if len(beat_matrix) < 3:
            return 0.0
            
        # Generate Median Template
        local_template = np.median(beat_matrix, axis=0)
        
        # Individual Beat Correlation
        correlations = []
        for beat in beat_matrix:
            r, _ = pearsonr(beat, local_template)
            
            if np.isnan(r):
                r = 0.0
            # Clip negative correlations to 0
            correlations.append(max(0.0, r))
            
        # Window Aggregation
        window_sqi = float(np.mean(correlations))
        
        return window_sqi
    
    def extract_fixed_window_fiducials(self, preprocessed, window_start: int, window_end: int):
        """
        Extracts heart rate from PPG within the given window.
        
        param self: PPG object
        param preprocessed: np.ndarray, preprocessed PPG signal data
        param window_start: int, index of window start
        param window_end: int, index of window end
        """

        sig = preprocessed[window_start:window_end]
        
        # Failsafe for empty windows
        if len(sig) == 0:
            return {f"{self.name}_HR": np.nan}
            
        # Calculate Power Spectral Density
        freqs, psd = welch(sig, fs=self.sampling_rate, nperseg=len(sig))
        
        # Define physiological human heartbeat range in Hz
        # 40 BPM = 0.67 Hz | 220 BPM = 3.67 Hz
        min_hz = 40.0 / 60.0
        max_hz = 180.0 / 60.0
        
        # Mask the frequencies to only our region of interest
        valid_idx = np.where((freqs >= min_hz) & (freqs <= max_hz))[0]
        
        if len(valid_idx) == 0:
            return {f"{self.name}_HR": np.nan}
            
        # Extract the relevant frequencies and their corresponding power
        valid_freqs = freqs[valid_idx]
        valid_psd = psd[valid_idx]
        
        # Find the frequency bin with the highest energy
        peak_freq = valid_freqs[np.argmax(valid_psd)]
        
        # Convert Hz back to Beats Per Minute (BPM)
        hr_bpm = float(peak_freq * 60.0)

        return {f"{self.name}_HR": hr_bpm}