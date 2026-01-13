from __dataset_processors import dataset
from __session_processors import session
from __signal_processors import physiological_signal
from __feature_processors import feature
import numpy as np



test_dataset = dataset(
    datasetName="TestDataset",
    sessionIds=["p22_1", "p28_2"],
    signals=[physiological_signal(name="ECG_downsample", sampling_rate=1000),
            physiological_signal(name="PPG_Finger_downsample", sampling_rate=1000)],
    data_directory="./data"
)
test_dataset.load_sessions(
    demographics_fp="./demographics.csv"
    )
# Now you can use dot notation! Use _22 for numeric session IDs
print(np.array(test_dataset.sessions.p22_1.signals.PPG_Finger_downsample.get_data()).shape)


