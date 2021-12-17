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


def rr_calulation(ppg_sig, fl=0.17, fh=0.35, o=5):
    fs = 25

    p1 = normalize(ppg_sig)
    ppg_1 = bessel_bandpass_filter(p1, fl, fh, fs=fs, order=o)
    ppg_11 = normalize(standardize(ppg_1))
    ppg_bpf = ppg_11

    peaks, _ = find_peaks(ppg_bpf, height=np.var(ppg_bpf), distance=10)
    #
    rr_bessel = len(peaks)

    return rr_bessel


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
        non_uniform = 0
        for a in range(len(peaks_all2) - 1):
            if len(rr_interval) > 0 and 399 < (peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs < 1800.0:
                rr_interval.append((peaks_all2[a + 1] - peaks_all2[a]) * 1000 / fs)
            elif len(rr_interval) > 0:
                rr_interval.append(rr_interval[-1])
                non_uniform += 1
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

    hr_pr_df = pd.DataFrame()
    hr_pr_df['timestamps'] = time_stamp
    hr_pr_df['Heart Rate Predicted'] = pd.Series(hr_diff).rolling(r).mean()

    final_pr = hr_pr_df.dropna()
    final_pr.iloc[:, 1:] = final_pr.iloc[:, 1:].astype('int64')

    hr_extracted = final_pr['Heart Rate Predicted'].to_numpy()
    for hr in hr_extracted:
        spo2_pred.append(spo2(hr))

    return final_pr, ppg_sig, ppg_bpf, rr_interval, hr_extracted, non_uniform, spo2_pred


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
    # print(ppg_json_array)
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
    loaded_json = json.loads(ppg_json_array)
    # print('loaded')
    # print(loaded_json)
    # print(type(loaded_json))
    # print(type(ppg_json_array))
    for ppg_data in loaded_json:
        # print('ppg')
        # print(ppg_data)
        ppg_sec = ppg_data['data']
        # print(ppg_sec)
        time_val.append(ppg_data['app_date'].split()[1])

        for j in range(2, len(ppg_sec), 3):
            ppg_bytes.append(decimal_to_binary(ppg_sec[j + 1]) + decimal_to_binary(ppg_sec[j]))
    ppg_sig = []
    for i in range(len(ppg_bytes)):
        ppg_sig.append(as_signed_big(ppg_bytes[i]))
    ppg_sig = np.asarray(ppg_sig)

    final_pr, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted, non_uniform, spo2_pred = ppg_plot_hr(
        ppg_sig, time_val, fl=1, fh=5, o=4, n=5, diff_max=10, r=5)
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
           "resp_rate": resp_rate, "rr_peak_intervals": t_diff_afib, 'a_Fib': afib_in, "tachycardia": tachy_in,
           "bradycardia": brady_in, "hr_extracted": hr_extracted.astype(int).tolist()}
    # print(res)
    return res

# data='[{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:12","data":[-7,0,20,-111,1,40,-111,2,51,-111,3,64,-111,4,80,-111,5,96,-111,6,113,-111,7,125,-111,8,-115,-111,9,-103,-111,10,-88,-111,11,-69,-111,12,-47,-111,13,-21,-111,14,9,-110,15,35,-110,16,75,-110,17,106,-110,18,-121,-110,19,-90,-110,20,-66,-110,21,-48,-110,22,-22,-110,23,1,-109,24,16,-15],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data":[-7,0,19,-109,1,10,-109,2,2,-109,3,-3,-110,4,2,-109,5,14,-109,6,24,-109,7,28,-109,8,34,-109,9,36,-109,10,45,-109,11,59,-109,12,78,-109,13,101,-109,14,-127,-109,15,-106,-109,16,-88,-109,17,-80,-109,18,-79,-109,19,-79,-109,20,-83,-109,21,-92,-109,22,-102,-109,23,-111,-109,24,-124,126],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:14","data":[-7,0,123,-109,1,114,-109,2,108,-109,3,100,-109,4,93,-109,5,81,-109,6,80,-109,7,79,-109,8,77,-109,9,73,-109,10,76,-109,11,71,-109,12,60,-109,13,62,-109,14,58,-109,15,47,-109,16,37,-109,17,17,-109,18,-5,-110,19,-15,-110,20,-20,-110,21,-19,-110,22,-18,-110,23,-10,-110,24,-11,-47],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:15","data":[-7,0,-10,-110,1,-5,-110,2,-3,-110,3,1,-109,4,5,-109,5,12,-109,6,23,-109,7,23,-109,8,22,-109,9,20,-109,10,21,-109,11,25,-109,12,24,-109,13,24,-109,14,21,-109,15,24,-109,16,26,-109,17,27,-109,18,28,-109,19,32,-109,20,38,-109,21,45,-109,22,45,-109,23,41,-109,24,39,14],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:16","data":[-7,0,37,-109,1,38,-109,2,37,-109,3,38,-109,4,39,-109,5,45,-109,6,52,-109,7,61,-109,8,68,-109,9,77,-109,10,83,-109,11,94,-109,12,100,-109,13,99,-109,14,98,-109,15,98,-109,16,105,-109,17,115,-109,18,119,-109,19,123,-109,20,127,-109,21,-121,-109,22,-114,-109,23,-102,-109,24,-93,-76],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:18","data":[-7,0,-82,-109,1,-72,-109,2,-69,-109,3,-73,-109,4,-74,-109,5,-75,-109,6,-70,-109,7,-64,-109,8,-61,-109,9,-53,-109,10,-44,-109,11,-36,-109,12,-27,-109,13,-19,-109,14,-14,-109,15,-8,-109,16,0,-108,17,6,-108,18,6,-108,19,8,-108,20,9,-108,21,9,-108,22,13,-108,23,15,-108,24,18,0],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,20,-108,1,28,-108,2,32,-108,3,37,-108,4,47,-108,5,57,-108,6,65,-108,7,76,-108,8,88,-108,9,98,-108,10,104,-108,11,109,-108,12,115,-108,13,123,-108,14,-126,-108,15,-119,-108,16,-111,-108,17,-103,-108,18,-89,-108,19,-80,-108,20,-70,-108,21,-59,-108,22,-56,-108,23,-56,-108,24,-51,84],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,20,-108,1,28,-108,2,32,-108,3,37,-108,4,47,-108,5,57,-108,6,65,-108,7,76,-108,8,88,-108,9,98,-108,10,104,-108,11,109,-108,12,115,-108,13,123,-108,14,-126,-108,15,-119,-108,16,-111,-108,17,-103,-108,18,-89,-108,19,-80,-108,20,-70,-108,21,-59,-108,22,-56,-108,23,-56,-108,24,-51,84],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
# '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:12","data":[-7,0,20,-111,1,40,-111,2,51,-111,3,64,-111,4,80,-111,5,96,-111,6,113,-111,7,125,-111,8,-115,-111,9,-103,-111,10,-88,-111,11,-69,-111,12,-47,-111,13,-21,-111,14,9,-110,15,35,-110,16,75,-110,17,106,-110,18,-121,-110,19,-90,-110,20,-66,-110,21,-48,-110,22,-22,-110,23,1,-109,24,16,-15],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data":[-7,0,19,-109,1,10,-109,2,2,-109,3,-3,-110,4,2,-109,5,14,-109,6,24,-109,7,28,-109,8,34,-109,9,36,-109,10,45,-109,11,59,-109,12,78,-109,13,101,-109,14,-127,-109,15,-106,-109,16,-88,-109,17,-80,-109,18,-79,-109,19,-79,-109,20,-83,-109,21,-92,-109,22,-102,-109,23,-111,-109,24,-124,126],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:14","data":[-7,0,123,-109,1,114,-109,2,108,-109,3,100,-109,4,93,-109,5,81,-109,6,80,-109,7,79,-109,8,77,-109,9,73,-109,10,76,-109,11,71,-109,12,60,-109,13,62,-109,14,58,-109,15,47,-109,16,37,-109,17,17,-109,18,-5,-110,19,-15,-110,20,-20,-110,21,-19,-110,22,-18,-110,23,-10,-110,24,-11,-47],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:15","data":[-7,0,-10,-110,1,-5,-110,2,-3,-110,3,1,-109,4,5,-109,5,12,-109,6,23,-109,7,23,-109,8,22,-109,9,20,-109,10,21,-109,11,25,-109,12,24,-109,13,24,-109,14,21,-109,15,24,-109,16,26,-109,17,27,-109,18,28,-109,19,32,-109,20,38,-109,21,45,-109,22,45,-109,23,41,-109,24,39,14],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:16","data":[-7,0,37,-109,1,38,-109,2,37,-109,3,38,-109,4,39,-109,5,45,-109,6,52,-109,7,61,-109,8,68,-109,9,77,-109,10,83,-109,11,94,-109,12,100,-109,13,99,-109,14,98,-109,15,98,-109,16,105,-109,17,115,-109,18,119,-109,19,123,-109,20,127,-109,21,-121,-109,22,-114,-109,23,-102,-109,24,-93,-76],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:18","data":[-7,0,-82,-109,1,-72,-109,2,-69,-109,3,-73,-109,4,-74,-109,5,-75,-109,6,-70,-109,7,-64,-109,8,-61,-109,9,-53,-109,10,-44,-109,11,-36,-109,12,-27,-109,13,-19,-109,14,-14,-109,15,-8,-109,16,0,-108,17,6,-108,18,6,-108,19,8,-108,20,9,-108,21,9,-108,22,13,-108,23,15,-108,24,18,0],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,20,-108,1,28,-108,2,32,-108,3,37,-108,4,47,-108,5,57,-108,6,65,-108,7,76,-108,8,88,-108,9,98,-108,10,104,-108,11,109,-108,12,115,-108,13,123,-108,14,-126,-108,15,-119,-108,16,-111,-108,17,-103,-108,18,-89,-108,19,-80,-108,20,-70,-108,21,-59,-108,22,-56,-108,23,-56,-108,24,-51,84],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,20,-108,1,28,-108,2,32,-108,3,37,-108,4,47,-108,5,57,-108,6,65,-108,7,76,-108,8,88,-108,9,98,-108,10,104,-108,11,109,-108,12,115,-108,13,123,-108,14,-126,-108,15,-119,-108,16,-111,-108,17,-103,-108,18,-89,-108,19,-80,-108,20,-70,-108,21,-59,-108,22,-56,-108,23,-56,-108,24,-51,84],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:19","data":[-7,0,-51,-108,1,-44,-108,2,-40,-108,3,-35,-108,4,-32,-108,5,-31,-108,6,-30,-108,7,-23,-108,8,-16,-108,9,-12,-108,10,-6,-108,11,2,-107,12,2,-107,13,4,-107,14,7,-107,15,14,-107,16,22,-107,17,32,-107,18,39,-107,19,44,-107,20,47,-107,21,51,-107,22,53,-107,23,56,-107,24,61,-124],"id":0},' \
#      '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:20","data":[-7,0,70,-107,1,78,-107,2,85,-107,3,85,-107,4,86,-107,5,90,-107,6,94,-107,7,90,-107,8,85,-107,9,81,-107,10,74,-107,11,71,-107,12,70,-107,13,70,-107,14,68,-107,15,70,-107,16,73,-107,17,77,-107,18,69,-107,19,63,-107,20,57,-107,21,55,-107,22,54,-107,23,51,-107,24,50,53],"id":0}]'
# res=ailments_stats(data)
# print(res)