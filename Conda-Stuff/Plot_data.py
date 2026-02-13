import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) > 1:
    file_list = sys.argv[1:]
    dataframes = [pd.read_csv(file) for file in file_list]

    colors = plt.cm.tab20(np.linspace(0, 1, len(dataframes)))

    for i, (df, color) in enumerate(zip(dataframes, colors)):
        plt.scatter(
            df['Epoch'],
            df['gpu_time'],
            color=color
        )


    all_values = []
    for df in dataframes:
        df['gpu_time'] = pd.to_numeric(df['gpu_time'], errors='coerce')
        values = df['gpu_time'].iloc[1:]  # skip first epoch
        all_values.append(values)

    all_values_concat = pd.concat(all_values)

    # Compute global average
    global_avg = all_values_concat.mean()
    plt.axhline(y=global_avg, color='black', linestyle='--', linewidth=2,
            label=f'Global Average={global_avg:.3f}')
    
    plt.xlabel("Epoch")
    plt.ylabel("GPU Time")
    plt.tight_layout()
    plt.legend()
    plt.savefig("8gpu graph.png")