

import matplotlib as mpl
import numpy as np

mpl.rc('font', size=20)
mpl.rc('figure', figsize=(20, 10))
import warnings

warnings.filterwarnings("ignore")
from bitstring import BitStream, BitArray
from scipy.signal import butter, lfilter
import scipy.signal as signal
import math
from scipy.signal import find_peaks


def DecimalToBinary(num):
    if num >= 0:
        dec = bin(num).lstrip('0b')

    if num < 0:
        dec = num.to_bytes(1, 'big', signed=True)
        dec = BitArray(bytes=dec).bin
    if len(dec) < 8:
        diff = 8 - len(dec)
        for i in range(diff):
            dec = '0' + dec
        # print(dec)
    return dec


def as_signed_big(binary_str):
    # This time, taking advantage of positional args and default values.
    as_bytes = int(binary_str, 2).to_bytes(2, 'big')
    return int.from_bytes(as_bytes, 'big', signed=True)


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


def high_pass_IIR(data, fh, samp_f, order):
    b, a = signal.butter(order, fh / (samp_f / 2), btype='high', output='ba')
    high_data = signal.lfilter(b, a, data)
    return high_data


fs = 20
n = 30 * 20


def to_decimal(acc):
    acc_Xind = []
    for j in range(acc.shape[0]):
        for i in range(1, 121, 6):
            acc_Xind.append(
                DecimalToBinary(acc.iloc[j, i + 1].item())[-1::-1] + DecimalToBinary(acc.iloc[j, i].item())[-1::-1])

    acc_X = []
    for i in range(len(acc_Xind)):
        acc_X.append(as_signed_big(acc_Xind[i]))
    acc_X = np.asarray(acc_X)

    acc_Yind = []
    for j in range(acc.shape[0]):
        for i in range(3, 121, 6):
            acc_Yind.append(
                DecimalToBinary(acc.iloc[j, i + 1].item())[-1::-1] + DecimalToBinary(acc.iloc[j, i].item())[-1::-1])

    acc_Y = []
    for i in range(len(acc_Yind)):
        acc_Y.append(as_signed_big(acc_Yind[i]))
    acc_Y = np.asarray(acc_Y)

    acc_Zind = []
    for j in range(acc.shape[0]):
        for i in range(5, 121, 6):
            acc_Zind.append(
                DecimalToBinary(acc.iloc[j, i + 1].item())[-1::-1] + DecimalToBinary(acc.iloc[j, i].item())[-1::-1])

    acc_Z = []
    for i in range(len(acc_Zind)):
        acc_Z.append(as_signed_big(acc_Zind[i]))
    acc_Z = np.asarray(acc_Z)

    for index_z in range(19, len(acc_Z), 20):
        acc_Z[index_z] = np.mean(acc_Z[index_z - 19: index_z - 1])

    return acc_X, acc_Y, acc_Z


def step_count(acc_X, acc_Y, acc_Z):
    a_x = acc_X * 18.3 / 128.0 / 1000.0 + 0.06
    a_y = acc_Y * 18.3 / 128.0 / 1000.0 + 0.06
    a_z = acc_Z * 18.3 / 128.0 / 1000.0 + 0.06

    x_abs = abs(a_x)
    y_abs = abs(a_y)
    z_abs = abs(a_z)
    svm = []

    for i in range(0, len(a_y)):
        svm.append(math.sqrt(x_abs[i] ** 2 + y_abs[i] * 2 + z_abs[i] ** 2))

    peaks, properties = find_peaks(svm, height=2, prominence=1)
    peaks_heights = properties['peak_heights']
    step = len(peaks)

    peaks_heights_4root = [(l * (float(1 / 4))) / 2 for l in peaks_heights]
    peaks_height_sum = np.sum(peaks_heights_4root)

    stride = float('%.2f' % ((np.mean(peaks_heights_4root) - 0.1) * 2))
    peaks_height_sum = float('%.2f' % (peaks_height_sum))

    if len(peaks) == 0 or len(peaks_heights) == 0:
        step, peaks_height_sum, stride = 0, 0, 0

    return step, peaks_height_sum, stride








