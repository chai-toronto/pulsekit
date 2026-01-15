class Feature:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.output = None

    def compute(self, *args, **kwargs):
        """placeholder method for computing the feature."""
        pass

    def save_outputs(self, filepath):
        """placeholder method for saving outputs."""
