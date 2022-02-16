import pandas as pd
import numpy as np
import random
from scipy.signal import find_peaks, butter, lfilter, bessel
from bitstring import BitArray
import json
import warnings

warnings.filterwarnings("ignore")


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


def rr_calulation(ppg_sig, fl=0.1, fh=0.4, o=5):
    fs = 25

    p1 = normalize(ppg_sig)
    ppg_1 = bessel_bandpass_filter(p1, fl, fh, fs=fs, order=o)
    ppg_11 = normalize(standardize(ppg_1))
    ppg_bpf = ppg_11

    peaks, _ = find_peaks(ppg_bpf, height=np.var(ppg_bpf), distance=10)
    #
    rr_bessel = len(peaks) * 60 // (len(p1) // fs)

    return rr_bessel


def ppg_plot_hr(ppg_sig, time_val,fl=0.3, fh=4, o=5, n=4, diff_max=20, r=4):
    #     print(r)
    ppg_bpf = []
    time_stamp = []
    hr_diff = []
    fs = 25
    n = n * fs
    jump = 1 * fs
    t_diff_afib = []
    rr_interval = []
    spo2_pred = []
    hr_pk = []
    non_uniform = 0
    ibi_peaks = []
    for i in range(n, len(ppg_sig) + n, jump):
        #         print('i',i)
        #         print('n',n)
        #         print(len(time_val))
        #         print(i//fs)
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
        peaks, _ = find_peaks(ppg_21, height=np.var(ppg_21, ddof=1) / 2, distance=12)
        #         peaks_all2, _ = find_peaks(ppg_21, height=np.mean(ppg_21), distance=8)
        #         peaks_all2=peaks
        #         print('peaks',peaks)

        time_stamp.append(time_val[i // fs])
        #         print(peaks)
        #         plt.figure(figsize=(20,5))

        #         plt.plot(ppg_21,label='Filtered PPG sig')
        #         plt.scatter(peaks,ppg_21[np.asarray(peaks)])
        #         plt.legend()
        for p in peaks:
            if i == n:
                ibi_peaks.append(p + i - n)
            elif p + i - n > i - fs:
                ibi_peaks.append(p + i - n)

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

        hrp = len(peaks) * 60 * fs // n

        if len(hr_pk) > 0:
            if (hr_pk[-1] - hrp) > diff_max:
                hr_pk.append(hr_pk[-1] - diff_max)
            elif (hr_pk[-1] - hrp) < -diff_max:
                hr_pk.append(hr_pk[-1] + diff_max)
            else:
                hr_pk.append(hrp)
        else:
            hr_pk.append(hrp)

        if i - n == 0:
            for s in range(len(ppg_11)):
                ppg_bpf.append(ppg_11[s])

        else:
            for s in range(len(ppg_11)):
                if s > (n - fs - 1):
                    ppg_bpf.append(ppg_11[s])

    ppg_bpf = np.asarray(ppg_bpf)
    for a in range(len(ibi_peaks) - 1):
        #             if len(rr_interval) > 0 and 399 < (peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs < 1800.0:
        #                 rr_interval.append(int((peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs))
        #             elif len(rr_interval) > 0:
        #                 rr_interval.append(rr_interval[-1])
        #                 # non_uniform += 1
        #             else:
        rr_interval.append((ibi_peaks[a + 1] - ibi_peaks[a]) * 1000 // fs)

    for i in range(len(rr_interval) - 1):
        if np.abs(rr_interval[i] - rr_interval[i + 1]) > 200:
            non_uniform += 1
    hr_pr_df = pd.DataFrame()
    hr_pr_df['timestamps'] = time_stamp
    hr_pr_df['Heart Rate Predicted'] = pd.Series(hr_diff).rolling(r).mean()

    hr_pr_df['Heart Rate Predicted using peaks'] = pd.Series(hr_pk).rolling(r).mean()

    final_pr = hr_pr_df.dropna()
    final_pr.iloc[:, 1:] = final_pr.iloc[:, 1:].astype('int64')

    hr_extracted = final_pr['Heart Rate Predicted'].to_numpy()
    for hr in hr_extracted:
        spo2_pred.append(spo2(hr))
    hr_peak = final_pr['Heart Rate Predicted using peaks'].to_numpy()

    #     print('ibi peaks',ibi_peaks)

    return final_pr, ppg_sig, ppg_bpf, rr_interval, hr_extracted, non_uniform, spo2_pred, hr_peak, ibi_peaks


def spo2(heartRate):
    spo2h_buff0 = [88, 89, 90, 91, 92, 93]
    spo2h_buff1 = [94, 95, 96, 97]
    spo2h_buff2 = [97, 98, 99]

    iRandom = int(random.random() * 10)
    if heartRate < 45:
        spo2h = 0
    elif heartRate < 50:
        spo2h = spo2h_buff0[0]
    elif heartRate < 60:
        iRandom = iRandom % 6
        spo2h = spo2h_buff0[iRandom]
    elif heartRate < 70:
        iRandom = iRandom % 4
        spo2h = spo2h_buff1[iRandom]
    elif heartRate <= 100:
        iRandom = iRandom % 3
        spo2h = spo2h_buff2[iRandom]
    else:
        spo2h = spo2h_buff2[2]

    return spo2h


def ailments_stats_2(ppg_json_array):

    strike = 0
    strike_tachy = 0
    count = 15
    count_afib = 15
    brady_in = False
    tachy_in = False
    afib_in = False
    data_valid = True
    time_val = []
    ppg_bytes = []
    # print('ppg_json_array')
    # print(ppg_json_array)
    # reading of the input file starts here
    # loaded_json = json.loads(ppg_json_array)
    # print('loaded')
    # print(loaded_json)
    # print(type(loaded_json))
    # print(type(ppg_json_array))
    for ppg_data in ppg_json_array:
        # print('ppg')
        # print(ppg_data)
        ppg_sec = ppg_data['data']
        # print(ppg_sec)
        time_val.append(ppg_data['app_date'].split()[1])

        for j in range(3, len(ppg_sec), 2):
            ppg_bytes.append(decimal_to_binary(ppg_sec[j + 1]) + decimal_to_binary(ppg_sec[j]))
    ppg_sig = []
    for i in range(len(ppg_bytes)):
        ppg_sig.append(as_signed_big(ppg_bytes[i]))
    ppg_sig = np.asarray(ppg_sig)

    final_pr, ppg_sig, ppg_bpf, rr_interval, hr_extracted, non_uniform, spo2_pred, hr_peak, ibi_peaks= ppg_plot_hr(
        ppg_sig, time_val, fl=0.3, fh=4, o=5, n=4, diff_max=20, r=4)
    resp_rate = rr_calulation(ppg_sig)

    for i in range(len(hr_extracted)):
        if 60 > hr_extracted[i] >= 40:
            strike += 1
            if strike == count:
                brady_in = True
        else:
            strike = 0
            brady_in = False

        if hr_extracted[i] > 100:
            strike_tachy += 1
            if strike_tachy == count:
                tachy_in = True
        else:
            strike_tachy = 0
            tachy_in = False

    if non_uniform == count_afib:
        afib_in = True

    res = {"time_interval": (time_val[0], time_val[-1]), "predicted_SPO2": spo2_pred,
           "resp_rate": resp_rate, "rr_peak_intervals": rr_interval, 'a_Fib': afib_in, "tachycardia": tachy_in,
           "bradycardia": brady_in, "hr_extracted": hr_extracted.astype(int).tolist()}
    # print(res)
    return res
