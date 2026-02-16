import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) > 1:
    file_list = sys.argv[1:]
    dataframes = [pd.read_csv(file) for file in file_list]

    sums = []
    for df in dataframes:
        df['epoch_time'] = pd.to_numeric(df['epoch_time'], errors='coerce')
        values = df['epoch_time']  
        total = values.sum()
        sums.append(total)

    # Compute global average
    average = sum(sums) / len(sums)
    print(average)