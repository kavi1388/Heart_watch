from ..PPG.custom_modules import *
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter,bessel


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

def rr_calulation(ppg_sig,fl=0.17,fh=0.35,o=5):

    fs=25

    p1=normalize(ppg_sig)
    ppg_1=bessel_bandpass_filter(p1, fl, fh, fs=fs, order=o)
    ppg_11=normalize(standardize(ppg_1))
    ppg_bpf=ppg_11

    peaks, _ = find_peaks(ppg_bpf,height=np.var(ppg_bpf),distance=10)
#
    rr_bessel=len(peaks)

    return rr_bessel