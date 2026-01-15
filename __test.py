from __dataset_processors import Dataset
from __session_processors import Session
from signal_processors.base import PhysiologicalSignal
from __feature_processors import feature
import numpy as np


test_dataset = Dataset(
    datasetName="TestDataset",
    sessionIds=["p22_1", "p28_2"],
    signals=[PhysiologicalSignal(name="ECG_downsample", sampling_rate=1000),
            PhysiologicalSignal(name="PPG_Finger_downsample", sampling_rate=1000)],
    data_directory="./data"
)

test_dataset.load_sessions(
    demographics_fp="./demographics.csv"
    )

# Now you can use dot notation! Use _22 for numeric session IDs
print(np.array(test_dataset.sessions.p22_1.signals.PPG_Finger_downsample.get_data()).shape)