from .base_feature import Feature
import numpy as np

class PWA(Feature):
    NAME = "pulse_wave_analysis"
    NECESSARY_SIGNALS = []
    POSSIBLE_SIGNALS = ["PPG_Finger", "PPG_Ear", "PPG_Wrist"]
    WINDOWING_TYPE = "cardiac_cycles"

    def compute(self, session, row_data):
        """
        Calculates Pulse Wave Analysis features from fiducials already in segments.csv
        Can only be run on cardiac cycle windowed data.
        """
        results = {}
        for signal_name in self.signals:
            try:
                if not self.check_quality(row_data, [signal_name]):
                    results.update({
                    f"{signal_name}_amp": np.nan,
                    f"{signal_name}_amp_norm": np.nan,
                    f"{signal_name}_rise_time": np.nan
                    })
                    continue

                signal = session.signals[signal_name]
                data = signal.data
                
                # peak and foot values
                peak_idx = int(row_data[f"{signal_name}_peak"])
                foot_idx = int(row_data[f"{signal_name}_foot"])
                peak_val = data[peak_idx]
                foot_val = data[foot_idx]

                raw_amp = peak_val - foot_val
                window_data = data[int(row_data['start']):int(row_data['end'])]
                norm_amp = raw_amp / np.mean(window_data)

                results.update({
                    f"{signal_name}_amp": raw_amp,
                    f"{signal_name}_amp_norm": norm_amp,
                    f"{signal_name}_rise_time": (peak_idx - foot_idx) / signal.sampling_rate
                })
            except Exception as e:
                print(f"Error computing PWA features for session {session.session_id}, window {row_data["window_id"]}: {e}")
        
        return results