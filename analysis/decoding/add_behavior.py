import os

import pandas as pd
import numpy as np

"""
edit data key csv to include behavioral data. 
"""

path_to_fmristim = ""
subject = "wooster"
path_to_data_key = ""

data_key = pd.read_csv(path_to_data_key)

# parse fmri-stim file structure to dict
behavior_dict = {}

scan_sessions = os.listdir(path_to_fmristim)

for scan_sess in scan_sessions:
    if subject not in scan_sess.lower():
        continue
    results_path = os.path.join(path_to_fmristim, scan_sess, "results", "SessionResults.csv")
    results = pd.read_csv(results_path)
    date = "".join(results["Session"].split("-"))
    imas = pd.unique(results["ima"])
    behavior_dict[date] = {}
    for ima in imas:
        behavior_dict[date][ima] = results[results["ima"] == ima]["correct"]





