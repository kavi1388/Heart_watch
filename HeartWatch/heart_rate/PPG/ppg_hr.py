from custom_modules import *
import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.signal import find_peaks


def ppg_plot_hr(ppg_sig, time_val, fl=0.4, fh=3.5, o=4, n=12, diff_max=4, r=12):
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

    # print('PPG Signal is {} seconds long'.format(len(ppg_sig) // fs))
    for i in range(n, len(ppg_sig), jump):

        p1 = (ppg_sig * 18.3 / 128.0 / 1000)[i - n:i]
        ppg_1 = butter_bandpass_filter(p1, fl, fh, fs=fs, order=o)

        ppg_11 = standardize(ppg_1)
        # Power spectrum

        dppg = []
        for v in range(1, len(ppg_11) - 1, 1):
            dppg.append(ppg_11[v + 1] - ppg_11[v - 1])
        ppg_21 = np.asarray(dppg)
        corr = correlate(ppg_11, ppg_11, mode='full')
        corr = corr[len(corr) // 2:]
        peaks, _ = find_peaks(ppg_21, height=np.var(ppg_21, ddof=1) / 3, distance=4)
        peaks_acf, _ = find_peaks(corr, height=0, distance=4)
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

    ppg_bpf = np.asarray(ppg_bpf)
    peaks_all2 = np.unique(np.asarray(np.sort(peaks_all)).astype(int))
    for a in range(len(peaks_all2) - 1):
        rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) / fs)
    # print('Number of Peaks Detected in PPG is {}'.format(len(peaks_all2)))
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
    #     print(final_pr)

    hr_extracted = final_pr['heart predicted by acf'].to_numpy()
    return final_pr, dppg_comb, ppg_sig, ppg_bpf, rr_interval, hr_extracted
