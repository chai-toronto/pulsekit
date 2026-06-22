import numpy as np
from PulseKit import *  
import os



dataset = Dataset(
    datasetName="TestDataset",
     sessionIds=["HC_3", "HC_8_2", "HC_10_2", "HFC_139", "HFC_146", "HFC_148"],
    signals=[
        ECG(name="ECG"),
        PPG(name="PPG_Finger"),
        PPG(name="PPG_Ear"),
        Stethoscope(name="Digital_Stethoscope"),
        Earbud(name="Earbud_Left"),
        Earbud(name="Earbud_Right"),
        Accelerometer(name="Accelerometer_X"),
        Accelerometer(name="Accelerometer_Y"),
        Accelerometer(name="Accelerometer_Z"),
        RespirationBelt(name="Respiration_Belt"),
        BP(name="Diastolic_BP"),
        BP(name="Systolic_BP"),
        BP(name="NIBP"),
        Hemodynamics(name="Cardiac_Output"),
        Hemodynamics(name="Stroke_Volume"),
        Hemodynamics(name="SVR"),
        Hemodynamics(name="Heart_Rate")
    ],
    data_directory=r"test_data"
)


dataset.save_dataset_to_numpy(
    save_dir="./debug_data",
    signal_list = ["ECG", "PPG_Finger", "PPG_Ear", "Digital_Stethoscope", "Earbud_Left", "Earbud_Right", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z", "Respiration_Belt", "Diastolic_BP", "Systolic_BP", "NIBP", "Cardiac_Output", "Stroke_Volume", "SVR", "Heart_Rate"], 
)

