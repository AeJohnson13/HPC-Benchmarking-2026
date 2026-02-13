import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("means.csv")


fig, ax = plt.subplots()

bars = ax.bar(df['Gpus'],
        df['training_time'])

# Add value labels on top
ax.bar_label(bars, fmt='%.2f Seconds')  # 2 decimal places

plt.xlabel("# of Gpus")
plt.ylabel("Average Full Training Time")
plt.tight_layout()
plt.savefig("mean graph.png")