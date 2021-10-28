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


def standardize(x):
    return (x - np.mean(x)) / np.var(x)


def moving_average(x, MA_SIZE):
    y = np.copy(x)
    for i in range(len(x) - MA_SIZE):
        y[i] = np.sum(x[i:i + MA_SIZE]) / MA_SIZE
    return y


def calc_hr(ir_data, MA_SIZE):
    """
    By detecting  peaks of PPG cycle and corresponding AC/DC
    of red/infra-red signal, the an_ratio for the SPO2 is computed.
    """
    # get dc mean
    ir_mean = int(np.mean(ir_data))
    # 25 samples per second (in algorithm.h)
    SAMPLE_FREQ = 25

    x = -1 * (np.array(ir_data) - ir_mean)

    # MA point moving average
    # x is np.array with int values, so automatically casted to int
    for i in range(x.shape[0] - MA_SIZE):
        x[i] = np.sum(x[i:i + MA_SIZE]) / MA_SIZE

    # calculate threshold
    n_th = int(np.mean(x))

    n_th = 0 if n_th < 0 else n_th  # min allowed
    n_th = 6 if n_th > 6 else n_th  # max allowed
    #     print(n_th)
    ir_valley_locs, n_peaks = find_peaks_new(x, len(x), np.var(x, ddof=1) / 2, n_th, len(x) * 4 // SAMPLE_FREQ)

    peak_interval_sum = 0
    if n_peaks >= 2:
        for i in range(1, n_peaks):
            peak_interval_sum += (ir_valley_locs[i] - ir_valley_locs[i - 1])
        peak_interval_sum = int(peak_interval_sum / (n_peaks - 1))
        hr = int(SAMPLE_FREQ * 60 / peak_interval_sum)
    else:
        hr = -999  # unable to calculate because # of peaks are too small

    return hr


def find_peaks_new(x, size, min_height, min_dist, max_num):
    """
    Find at most MAX_NUM peaks above MIN_HEIGHT separated by at least MIN_DISTANCE
    """
    ir_valley_locs, n_peaks = find_peaks_above_min_height(x, size, min_height, max_num)
    ir_valley_locs, n_peaks = remove_close_peaks(n_peaks, ir_valley_locs, x, min_dist)

    n_peaks = min([n_peaks, max_num])

    return ir_valley_locs, n_peaks


def find_peaks_above_min_height(x, size, min_height, max_num):
    """
    Find all peaks above MIN_HEIGHT
    """

    i = 0
    n_peaks = 0
    ir_valley_locs = []  # [0 for i in range(max_num)]
    while i < size - 1:
        if x[i] > min_height and x[i] > x[i - 1]:  # find the left edge of potential peaks
            n_width = 1
            # original condition i+n_width < size may cause IndexError
            # so I changed the condition to i+n_width < size - 1
            while i + n_width < size - 1 and x[i] == x[i + n_width]:  # find flat peaks
                n_width += 1
            if x[i] > x[i + n_width] and n_peaks < max_num:  # find the right edge of peaks

                ir_valley_locs.append(i)
                n_peaks += 1  # original uses post increment
                i += n_width + 1
            else:
                i += n_width
        else:
            i += 1

    return ir_valley_locs, n_peaks


def remove_close_peaks(n_peaks, ir_valley_locs, x, min_dist):
    """
    Remove peaks separated by less than MIN_DISTANCE
    """
    # should be equal to maxim_sort_indices_descend
    # order peaks from large to small
    # should ignore index:0
    sorted_indices = sorted(ir_valley_locs, key=lambda im: x[im])
    sorted_indices.reverse()

    # this "for" loop expression does not check finish condition
    # for i in range(-1, n_peaks):
    i = -1
    while i < n_peaks:
        old_n_peaks = n_peaks
        n_peaks = i + 1
        # this "for" loop expression does not check finish condition
        # for j in (i + 1, old_n_peaks):
        j = i + 1
        while j < old_n_peaks:
            n_dist = (sorted_indices[j] - sorted_indices[i]) if i != -1 else (sorted_indices[j] + 1)
            if n_dist > min_dist or n_dist < -1 * min_dist:
                sorted_indices[n_peaks] = sorted_indices[j]
                n_peaks += 1  # original uses post increment
            j += 1
        i += 1

    sorted_indices[:n_peaks] = sorted(sorted_indices[:n_peaks])
    return sorted_indices, n_peaks
