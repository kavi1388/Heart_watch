from ..PPG.custom_modules import *
import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def ppg_plot_hr(ppg_sig, time_val, fl=0.4, fh=3.5, o=4, n=10, diff_max=4, r=1):
    ppg_bpf = []
    peaks_all = []
    time_stamp = []
    hr_pred = []
    hr_diff = []
    hr_calc = []
    hr_acf = []
    hr_calc2 = []
    fs = 25
    n = n * fs
    jump = 1 * fs
    t_diff_afib = []
    dppg_comb = []
    rr_interval = []

    for i in range(n, len(ppg_sig), jump):

        p1 = (ppg_sig * 18.3 / 128.0 / 1000)[i - n:i]
        ppg_1 = butter_bandpass_filter(p1, fl, fh, fs=fs, order=o)


        ppg_11 = standardize(ppg_1)
        # Power spectrum

        dppg = []
        for v in range(1, len(ppg_11) - 1, 1):
            dppg.append(ppg_11[v + 1] - ppg_11[v - 1])
        # ppg_21 = np.asarray(dppg)
        ppg_21=ppg_11
        corr = correlate(ppg_11, ppg_11, mode='full')
        corr = corr[len(corr) // 2:]
        c=[]
        for p in ppg_21:
            if p<0:
                c.append(p/20)
            else:
                c.append(p*20)
        peaks, _ = find_peaks(c, height=np.var(ppg_21, ddof=1) / 3, distance=10)

        peaks_acf, _ = find_peaks(corr, height=np.min(corr), distance=4)
        time_stamp.append(time_val[i // fs])

        dppg_comb.append(ppg_21)

        for pk in range(len(peaks)):
            pk0 = peaks[pk] + i - n
            peaks_all.append(pk0)

        t_diff = []
        for a in range(len(peaks) - 1):
            t_diff.append((peaks[a + 1] - peaks[a]) / fs)
            t_diff_afib.append(t_diff[-1])

        t_diff = np.asarray(t_diff)
        hrp = len(peaks) * 60 // (n // fs)
        hrd = 60 // t_diff.mean()

        hr_pred.append(hrp)
        hrcalc = calc_hr(ppg_21, 4)
        hrcalc2 = calc_hr(ppg_21, 6)
        hracp = len(peaks_acf) * 60 // (n // fs)

        if len(hr_pred) > 0:
            if (hr_pred[-1] - hrp) > diff_max:
                hr_pred.append(hr_pred[-1] - diff_max)
            elif (hr_pred[-1] - hrp) < -diff_max:
                hr_pred.append(hr_pred[-1] + diff_max)
            else:
                hr_pred.append(hrp)
        else:
            hr_pred.append(hrp)
        if len(hr_calc) > 0:
            if (hr_calc[-1] - hrcalc) > diff_max:
                hr_calc.append(hr_calc[-1] - diff_max)
            elif (hr_calc[-1] - hrcalc) < -diff_max:
                hr_calc.append(hr_calc[-1] + diff_max)
            else:
                hr_calc.append(hrcalc)
        else:
            hr_calc.append(hrcalc)

        if len(hr_calc2) > 0:
            if (hr_calc2[-1] - hrcalc2) > diff_max:
                hr_calc2.append(hr_calc2[-1] - diff_max)
            elif (hr_calc2[-1] - hrcalc2) < -diff_max:
                hr_calc2.append(hr_calc2[-1] + diff_max)
            else:
                hr_calc2.append(hrcalc2)
        else:
            hr_calc2.append(hrcalc2)
        if len(hr_acf) > 0:
            if (hr_acf[-1] - hracp) > diff_max:
                hr_acf.append(hr_acf[-1] - diff_max)
            elif (hr_acf[-1] - hracp) < -diff_max:
                hr_acf.append(hr_acf[-1] + diff_max)
            else:
                hr_acf.append(hracp)
        else:
            hr_acf.append(hracp)

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
    p1 = (ppg_sig * 18.3 / 128.0 / 1000)
    ppg_bpf = butter_bandpass_filter(p1, 0.4, 3.5, fs=fs, order=4)
    ppg_bpf = np.asarray(ppg_bpf)
    peaks, _ = find_peaks(ppg_bpf*20, height=np.var(ppg_bpf, ddof=1) / 3, distance=10)
    peaks_all2 = peaks
    non_uniform=0
    for a in range(len(peaks_all2) - 1):
        if len(rr_interval)>0 and 399 < (peaks_all2[a + 1] - peaks_all2[a]) *1000 / fs <1800.0:
            rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) *1000 / fs)
        elif len(rr_interval)>0:
            rr_interval.append(rr_interval[-1])
            non_uniform+=1
        else:
            rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs)

    hr_pr_df = pd.DataFrame()
    hr_pr_df['timestamps'] = time_stamp
    hr_pr_df['heart predicted by peak detection'] = pd.Series(hr_pred).rolling(r).median()
    hr_pr_df['heart predicted by peak time diff'] = pd.Series(hr_diff).rolling(r).median()
    hr_pr_df['heart predicted by acf'] = pd.Series(hr_acf).rolling(r).median()
    hr_pr_df['heart predicted by maxim code'] = pd.Series(hr_calc).rolling(r).median()
    hr_pr_df['heart predicted by maxim code ver2'] = pd.Series(hr_calc2).rolling(r).median()
    hr_avg = []
    for index in range(len(hr_calc)):
        hr_avg.append((hr_calc[index] + hr_acf[index]) // 2)
    hr_3_avg = []
    for index in range(len(hr_calc)):
        hr_3_avg.append((hr_calc[index] + hr_acf[index] + hr_calc2[index]) // 3)
    hr_all_avg = []
    for index in range(len(hr_calc)):
        hr_all_avg.append((hr_calc[index] + hr_acf[index] + hr_calc2[index] + hr_pred[index] + hr_diff[index]) // 5)
    hr_pr_df['heart predicted avg of acf and maxim'] = pd.Series(hr_avg).rolling(r).median()
    hr_pr_df['heart predicted avg of acf,maxim and maxim2'] = pd.Series(hr_3_avg).rolling(r).median()
    hr_pr_df['heart predicted avg of 5 ways'] = pd.Series(hr_all_avg).rolling(r).median()

    final_pr = hr_pr_df.dropna()
    final_pr.iloc[:, 1:] = final_pr.iloc[:, 1:].astype('int64')

    hr_extracted = final_pr['heart predicted by acf'].to_numpy()

    return final_pr,  ppg_sig, ppg_bpf, rr_interval, hr_extracted, peaks_all2, non_uniform