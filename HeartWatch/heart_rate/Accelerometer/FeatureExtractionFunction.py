import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy import signal
import statistics as sts
from scipy.stats import kurtosis, skew
#%matplotlib inline
# plt.style.use('ggplot')
from scipy.stats import iqr
from scipy.signal import butter, lfilter
import math


def sma(x):
    sma_val = 0
    for i in x:
        sma_val += abs(i)
    return sma_val


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size)


def peak_to_peak(s):
    max_val = max(s)
    min_val = min(s)
    return max_val - min_val


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y


def normalize(x):
    return x / np.max(np.abs(x))


def scale(X):
    return (X - np.mean(X)) / np.std(X)


def low_pass_IIR(data, fl, samp_f, order):
    b, a = signal.butter(order, fl / (samp_f / 2), btype='low', output='ba')
    low_data = signal.lfilter(b, a, data)
    return low_data


def filter(acc):
    fs = 20
    a_x = acc['ACC_X (in g)']
    a_gx = low_pass_IIR(a_x, 0.19, fs, 3)
    a_ux = a_x - a_gx
    a_y = acc['ACC_Y (in g)']
    a_gy = low_pass_IIR(a_y, 0.19, fs, 3)
    a_uy = a_y - a_gy
    a_z = acc['ACC_Z (in g)']
    a_gz = low_pass_IIR(a_z, 0.19, fs, 3)
    a_uz = a_z - a_gz

    return a_ux, a_uy, a_uz


def svm(x, y, z):
    svm_val = 0
    x = x.tolist()
    y = y.tolist()
    z = z.tolist()
    for i in range(0, len(x)):
        svm_val += math.sqrt(abs(x[i]) ** 2 + abs(y[i]) ** 2 + abs(z[i]) * 2)
    return svm_val


def corr(a, b, c):
    non_corr_count = 0
    corr1 = np.corrcoef(a, b)
    corr2 = np.corrcoef(a, c)
    corr3 = np.corrcoef(b, c)
    if math.isnan(corr1[0, 1]):
        non_corr_count += 1
    if math.isnan(corr2[0, 1]):
        non_corr_count += 1
    if math.isnan(corr3[0, 1]):
        non_corr_count += 1

    return non_corr_count


def extract_features(data, window_size=20):
    col_names = ['max_x', 'min_x', 'mean_x', 'std_x', 'median_x', 'skew_x', 'kurtosis_x', 'iqr_x', 'sma_x', 'p2p_x', \
                 'max_ux', 'min_ux', 'mean_ux', 'std_ux', 'median_ux', \
                 'max_y', 'min_y', 'mean_y', 'std_y', 'median_y', 'skew_y', 'kurtosis_y', 'iqr_y', 'sma_y', 'p2p_y', \
                 'max_uy', 'min_uy', 'mean_uy', 'std_uy', 'median_uy',
                 'max_z', 'min_z', 'mean_z', 'std_z', 'median_z', 'skew_z', 'kurtosis_z', 'iqr_z', 'sma_z', 'p2p_z', \
                 'max_uz', 'min_uz', 'mean_uz', 'std_uz', 'median_uz', 'svm', 'nan_corr_count']
    extractFeature_df = pd.DataFrame(columns=col_names)

    for (start, end) in windows(data['timestep'], window_size):
        extract_list = []
        labels = np.empty((0))
        a_ux, a_uy, a_uz = filter(data[start:end])

        x = data["ACC_X (in g)"][start:end]
        max_x, min_x, mean_x, std_x, median_x = max(x), min(x), np.mean(x), np.std(x), sts.median(x)
        skew_x, kurtosis_x, iqr_x = skew(x), kurtosis(x), iqr(x)
        sma_x = sma(x)
        p2p_x = peak_to_peak(x)

        max_ux, min_ux, mean_ux, std_ux, median_ux = max(a_ux), min(a_ux), np.mean(a_ux), np.std(a_ux), sts.median(a_ux)
        extract_list.extend((max_x, min_x, mean_x, std_x, median_x, skew_x, kurtosis_x, iqr_x, sma_x, p2p_x, max_ux,
                             min_ux, mean_ux, std_ux, median_ux))

        y = data["ACC_Y (in g)"][start:end]
        max_y, min_y, mean_y, std_y, median_y = max(y), min(y), np.mean(y), np.std(y), sts.median(y)
        skew_y, kurtosis_y, iqr_y = skew(y), kurtosis(y), iqr(y)
        sma_y = sma(y)
        p2p_y = peak_to_peak(y)
        max_uy, min_uy, mean_uy, std_uy, median_uy = max(a_uy), min(a_uy), np.mean(a_uy), np.std(a_uy), sts.median(a_uy)
        extract_list.extend((max_y, min_y, mean_y, std_y, median_y, skew_y, kurtosis_y, iqr_y, sma_y, p2p_y, max_uy,
                             min_uy, mean_uy, std_uy, median_uy))
        # print(extract_list)

        z = data["ACC_Z (in g)"][start:end]
        max_z, min_z, mean_z, std_z, median_z = max(z), min(z), np.mean(z), np.std(z), sts.median(z)
        skew_z, kurtosis_z, iqr_z = skew(z), kurtosis(z), iqr(z)
        sma_z = sma(z)
        p2p_z = peak_to_peak(z)
        max_uz, min_uz, mean_uz, std_uz, median_uz = max(a_uz), min(a_uz), np.mean(a_uz), np.std(a_uz), sts.median(a_uz)
        extract_list.extend((max_z, min_z, mean_z, std_z, median_z, skew_z, kurtosis_z, iqr_z, sma_z, p2p_z, max_uz,
                             min_uz, mean_uz, std_uz, median_uz))
        svm_val = svm(x, y, z)
        nan_corr_count = corr(x, y, z)
        extract_list.extend((svm_val, nan_corr_count))

        # extract_list.append(label)
        df_length = len(extractFeature_df)
        extractFeature_df.loc[df_length] = extract_list
        # print(extractFeature_df)

    return extractFeature_df

