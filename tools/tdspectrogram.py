import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import spectrogram
import os

def spec(csv_file):
    df = pd.read_csv(csv_file)
    time = df.iloc[:, 0].values
    frequency = df.iloc[:, 1].values
    fs = 1 / (time[1] - time[0])

    # Calculate spectrogram with fixed size
    nperseg = 447  # Number of DFT points
    window = scipy.signal.windows.blackman(nperseg)  # Blackman window
    noverlap = nperseg - 1 - 224  # Overlap length to set the width to 224
    f, t, Sxx = spectrogram(frequency, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    # Take natural log of the magnitude square of each spectrogram
    Sxx = np.log(np.abs(Sxx)**2 + 1e-10)

    filename = os.path.splitext(os.path.basename(csv_file))[0]

    plt.figure(figsize=(7/3, 7/3), facecolor='none', dpi=96)
    plt.pcolormesh(t, f, Sxx)
    plt.axis('off')
    output_image_path = f"temp_files/{csv_file}.png"
    plt.savefig(output_image_path, transparent=True, bbox_inches=0, pad_inches=0)
    plt.close()