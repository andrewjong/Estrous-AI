import pandas as pd
import os
from glob import glob

for prediction_file in glob("experiments/**/*predictions.csv", recursive=True):
    df = pd.read_csv(prediction_file)
    df['correct'] = df['label'] == df['predicted']
