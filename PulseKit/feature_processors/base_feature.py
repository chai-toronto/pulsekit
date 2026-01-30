class Feature:
    '''
    Class to store important data for the calculation of a feature and how to compute this feature.

    Default configurations (to be overridden by subclasses):
    - name: None
    - necessary_signals: []
    - possible_signals: []
    - windowing_type: None

    Attributes:
    - dataset: the dataset to calculate the feature across
    - signals: list of signals passed in to calculate feature with
    - apply_quality: boolean to determine whether or not to apply quality masks
    - output_name: file name to output to
    '''
    NAME = None
    NECESSARY_SIGNALS = []
    POSSIBLE_SIGNALS = []
    WINDOWING_TYPE = None

    def __init__(self, dataset, signals, output_name="features", apply_quality=True):
        self.dataset = dataset
        self.signals = signals
        self.output_name = output_name
        self.apply_quality = apply_quality
        
        # Run validations automatically
        self._validate_feature()

    def _validate_feature(self):
        """Common logic to ensure subclasses meet requirements."""
        for sig in self.NECESSARY_SIGNALS:
            if sig not in self.signals:
                raise AttributeError(f"Required signal '{sig}' not found for {self.NAME}")
            
        if self.dataset.windowing_type != self.WINDOWING_TYPE:
            raise AttributeError(
                f"Dataset windowing ({self.dataset.windowing_type}) "
                f"does not match {self.NAME} requirements ({self.WINDOWING_TYPE})."
            )
        
        for sig in self.signals:
            if sig not in self.POSSIBLE_SIGNALS:
                raise AttributeError(f"Signal '{sig}' is not valid for {self.NAME} feature.")

    def check_quality(self, row_data, signal_names):
        """Utility to check quality masks for a list of signals."""
        if not self.apply_quality:
            return True
        return all(row_data.get(f"{sig}_quality", True) for sig in signal_names)

    def compute(self, *args, **kwargs):
        """placeholder method for computing the feature."""
        pass

    def save_outputs(self, filepath):
        """placeholder method for saving outputs."""
        pass
