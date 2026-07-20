from .base_feature import Feature
import numpy as np

class PTT(Feature):
    NAME = "pulse_transit_time"
    NECESSARY_SIGNALS = []
    
    PROXIMAL_SIGNALS = ["Digital_Stethoscope", "ECG", "Accelerometer_Z"]
    DISTAL_SIGNALS = ["PPG_Finger", "PPG_Ear", "Earbud_Left", "Earbud_Right"]
    
    POSSIBLE_SIGNALS = PROXIMAL_SIGNALS + DISTAL_SIGNALS
    WINDOWING_TYPE = "cardiac_cycles"

    PHYSIOLOGICAL_BOUNDS = {
        ("ECG", "finger"): (0, 500),
        ("ECG", "ear"): (0, 500),
        ("ECG", "earbud"): (0, 500),
        ("Digital_Stethoscope", "finger"): (0, 500),
        ("Digital_Stethoscope", "ear"): (0, 500),
        ("Digital_Stethoscope", "earbud"): (0, 500),
        ("Accelerometer_Z", "finger"): (0, 500),
        ("Accelerometer_Z", "ear"): (0, 500),
        ("Accelerometer_Z", "earbud"): (0, 500),
    }

    def __init__(self, dataset, signals, output_name="features", apply_quality=True):
        self.active_proximal = []
        self.active_distal = []
        
        super().__init__(dataset, signals, output_name, apply_quality)

    def _validate_feature(self):
        """Overrides base validation to ensure at least one proximal signal is present."""
        super()._validate_feature()
        
        # Custom logic: Ensure at least one proximal signal exists
        self.active_proximal = [s for s in self.signals if s in self.PROXIMAL_SIGNALS]
        if not self.active_proximal:
            raise AttributeError(
                f"{self.NAME} requires at least one proximal signal: {self.PROXIMAL_SIGNALS}"
            )
        
        # Identify active distal signals
        self.active_distal = [s for s in self.signals if s in self.DISTAL_SIGNALS]
        if not self.active_distal:
             raise AttributeError(
                f"{self.NAME} requires at least one distal signal: {self.DISTAL_SIGNALS}"
            )

    def _get_proximal_fiducial(self, signal_name, row_data):
        """Helper to grab the correct fiducial based on signal type."""
        if signal_name == "ECG":
            return row_data.get("ECG_R_peak")
        elif signal_name == "Digital_Stethoscope":
            return row_data.get("Digital_Stethoscope_s1")
        elif signal_name == "Accelerometer_Z":
            return row_data.get("Accelerometer_Z_AO")
        return None
    
    def _get_distal_category(self, distal_name):
        name = distal_name.lower()
        if "finger" in name: return "finger"
        if "wrist" in name: return "wrist"
        if "earbud" in name: return "earbud"
        if "ear" in name: return "ear"
        return None

    def compute(self, session, row_data):
        """
        Calculates PTT for every proximal-distal pair.
        Proximal fiducials: ECG (R-peak), Steth (s1), Accelerometer_Z (AO).
        Distal fiducials: PPG (foot), Earbud (a).
        """
        results = {}

        for prox in self.active_proximal:
            prox_val = self._get_proximal_fiducial(prox, row_data)
            
            for distal in self.active_distal:
                # 1. Base Quality Check
                if not self.check_quality(row_data, [prox, distal]):
                    results[f"PTT_{prox}_{distal}"] = np.nan
                    continue

                # 2. Extract Distal Fiducial
                dist_val = row_data.get(f"{distal}_foot") if "ppg" in distal.lower() else row_data.get(f"{distal}_a")

                if not np.isnan(prox_val) and not np.isnan(dist_val):
                    ptt_value = int(dist_val) - int(prox_val)
                    
                    # 3. APPLY PHYSIOLOGICAL FILTERING
                    dist_cat = self._get_distal_category(distal)
                    bounds = self.PHYSIOLOGICAL_BOUNDS.get((prox, dist_cat))
                    
                    if bounds and not (bounds[0] <= ptt_value <= bounds[1]):
                        results[f"PTT_{prox}_{distal}"] = np.nan
                    else:
                        results[f"PTT_{prox}_{distal}"] = ptt_value
                else:
                    results[f"PTT_{prox}_{distal}"] = np.nan
        
        return results