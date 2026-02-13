import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    file_list = [sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]
    df = pd.read_csv(file_list[0])
    df1 = pd.read_csv(file_list[1])
    df2 = pd.read_csv(file_list[2])
    df3 = pd.read_csv(file_list[3])
    df4 = pd.read_csv(file_list[4])
    df5 = pd.read_csv(file_list[5])
    plt.scatter( df['Epoch'], df['gpu_time'], color='red', label='Gpu 1')
    plt.scatter( df1['Epoch'], df1['gpu_time'], color='blue', label='Gpu 2')
    plt.scatter( df2['Epoch'], df2['gpu_time'], color='red')
    plt.scatter( df3['Epoch'], df3['gpu_time'], color='blue')
    plt.scatter( df4['Epoch'], df4['gpu_time'], color='red')
    plt.scatter( df5['Epoch'], df5['gpu_time'], color='blue')

    plt.legend()
    plt.savefig("test.png")