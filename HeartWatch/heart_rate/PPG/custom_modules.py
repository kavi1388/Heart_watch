from bitstring import BitArray
from scipy.signal import butter, lfilter
import numpy as np


def decimal_to_binary(num):
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


def bessel_bandpass(lowcut=0.13, highcut=0.48, fs=25, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = bessel(order, [low, high], btype='band')
    return b, a


def bessel_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = bessel_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize(x):
    return x / np.max(np.abs(x))


def standardize(x):
    return (x - np.mean(x)) / np.var(x)
