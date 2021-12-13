from ..PPG.custom_modules import *
import numpy as np
from scipy.signal import find_peaks


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