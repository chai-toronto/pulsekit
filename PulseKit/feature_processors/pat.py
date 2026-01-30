from .base_feature import Feature
import numpy as np

class PAT(Feature):
    NAME = "pulse_arrival_time"
    NECESSARY_SIGNALS = ["ECG_downsample"]
    POSSIBLE_SIGNALS = ["PPG_Finger_downsample", "PPG_Ear_downsample", "PPG_Wrist_downsample", "Earbud_Left_downsample", "Earbud_Right_downsample", "ECG_downsample"]
    WINDOWING_TYPE = "cardiac_cycles"

    def compute(self, session, row_data):
            """
            Calculates PAT by comparing timing between two signals.
            PPG: foot of wave will be referenced against ECG R-peak.
            Earbud: 'a' point will be referenced against ECG R-peak.
            """
            
            results = {}
            ecg_peak = int(row_data["ECG_downsample_R_peak"])

            for signal_name in self.signals:
                if signal_name == "ECG_downsample":
                    continue

                if not self.check_quality(row_data, [signal_name, "ECG_downsample"]):
                    results[f"PAT_{signal_name}"] = np.nan
                    continue

                if "ppg" in signal_name.lower():
                    fiducial = row_data.get(f"{signal_name}_foot")
                elif "earbud" in signal_name.lower():
                    fiducial = row_data.get(f"{signal_name}_a")
                
                results[f"PAT_{signal_name}"] = (int(fiducial) - ecg_peak)
            
            return results