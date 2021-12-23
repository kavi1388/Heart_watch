from ..PPG.custom_modules import *
from ..PPG.spo2_from_ppg import *
import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def ppg_plot_hr(ppg_sig, time_val, fl=1, fh=5, o=4, n=5, diff_max=4, r=1):
    ppg_bpf = []
    time_stamp = []
    hr_diff = []
    fs = 25
    n = n * fs
    jump = 1 * fs
    t_diff_afib = []
    rr_interval = []
    spo2_pred = []
    non_uniform = 0
    for i in range(n, len(ppg_sig) + n, jump):

        if i > len(ppg_sig) - 1:
            break

        p1 = (ppg_sig * 18.3 / 128.0 / 1000)[i - n:i]
        ppg_1 = butter_bandpass_filter(p1, fl, fh, fs=fs, order=o)

        ppg_11 = normalize(standardize(ppg_1))

        ppg_21 = ppg_11

        c = []
        for p in ppg_21:
            if p < 0:
                c.append(p / 20)
            else:
                c.append(p * 20)
        peaks, _ = find_peaks(c, height=np.var(ppg_21, ddof=1) / 3, distance=11)
        peaks_all2, _ = find_peaks(ppg_21, height=np.mean(ppg_21), distance=11)

        for a in range(len(peaks_all2) - 1):
            if len(rr_interval) > 0 and 399 < (peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs < 1800.0:
                rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs)
            elif len(rr_interval) > 0:
                rr_interval.append(rr_interval[-1])
                # non_uniform += 1
            else:
                rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs)
        time_stamp.append(time_val[i // fs])

        t_diff = []
        for a in range(len(peaks) - 1):
            t_diff.append((peaks[a + 1] - peaks[a]) / fs)
            t_diff_afib.append(t_diff[-1])

        t_diff = np.asarray(t_diff)

        hrd = 60 // t_diff.mean()

        if len(hr_diff) > 0:
            if (hr_diff[-1] - hrd) > diff_max:
                hr_diff.append(hr_diff[-1] - diff_max)
            elif (hr_diff[-1] - hrd) < -diff_max:
                hr_diff.append(hr_diff[-1] + diff_max)
            else:
                hr_diff.append(hrd)
        else:
            hr_diff.append(hrd)

        if i - n == 0:
            for s in range(len(ppg_11)):
                ppg_bpf.append(ppg_11[s])

        else:
            for s in range(len(ppg_11)):
                if s > (n - fs - 1):
                    ppg_bpf.append(ppg_11[s])

    ppg_bpf = np.asarray(ppg_bpf)
    for i in range(len(rr_interval) - 1):
        if np.abs(rr_interval[i] - rr_interval[i + 1]) > 200:
            non_uniform += 1
    hr_pr_df = pd.DataFrame()
    hr_pr_df['timestamps'] = time_stamp
    hr_pr_df['Heart Rate Predicted'] = pd.Series(hr_diff).rolling(r).mean()

    final_pr = hr_pr_df.dropna()
    final_pr.iloc[:, 1:] = final_pr.iloc[:, 1:].astype('int64')

    hr_extracted = final_pr['Heart Rate Predicted'].to_numpy()
    for hr in hr_extracted:
        spo2_pred.append(spo2(hr))

    return final_pr, ppg_sig, ppg_bpf, rr_interval, hr_extracted, non_uniform, spo2_pred





