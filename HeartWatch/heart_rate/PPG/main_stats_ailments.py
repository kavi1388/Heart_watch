import matplotlib.pyplot as plt

from ppg_hr import *
from ..PPG.custom_modules import *
import datetime
import time
import json
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def ailments_stats(ppg_list):

    strike = 0
    strike_tachy = 0
    strike_afib = 0
    count = 20
    count_afib = 10
    brady_in= False
    tachy_in = False
    afib_in = False
    data_valid = True
    ppg_bytes = []
    time_val = []
    # reading of the input file starts here
    for ind in range(len(ppg_list)):
        a = ppg_list[ind]
        ppg_data = json.loads(a)
        ppg_sec = ppg_data['data']
        time_val.append(ppg_data['app_date'].split()[1])
        for j in range(2, len(ppg_sec), 3):
            ppg_bytes.append(decimal_to_binary(ppg_sec[j + 1]) + decimal_to_binary(ppg_sec[j]))
    ppg_sig = []
    for i in range(len(ppg_bytes)):
        ppg_sig.append(as_signed_big(ppg_bytes[i]))
    ppg_sig = np.asarray(ppg_sig)

    time_step_v = []
    for i in range(len(time_val)):

        x = time.strptime(time_val[i].split(',')[0], '%H:%M:%S')
        time_step_v.append(datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds())

        if len(time_step_v) > 2:
            if time_step_v[-2] - time_step_v[-1] > 120:
                data_valid = False
    if data_valid:
        final_pr, ppg_21, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted, peaks_all2,non_uniform = ppg_plot_hr(ppg_sig, time_val, fl=0.2, fh=3.5, o=4, n=6, diff_max=4, r=1)

        for i in range(len(hr_extracted)):
            if 60 > hr_extracted[i] >= 40:
                strike += 1
            else:
                strike = 0

            if strike == count:
                # print('Patient has Sinus Bradycardia')
                brady_in = True

                # One API call for Bradycardia

            if 100 < hr_extracted[i] <= 130:
                strike_tachy += 1
            else:
                strike_tachy = 0

            if strike_tachy == count:
                # print('Patient has Sinus Tachycardia')
                tachy_in = True

                # One API call for Tachycardia

        # for i in range(len(t_diff_afib) - 1):
        #     if t_diff_afib[i + 1] - t_diff_afib[i] > 10:
        #         strike_afib += 1
        #     else:
        #         strike_afib = 0
        if non_uniform == count_afib:
                # print('Patient has Atrial Fibrillation')
            afib_in = True

        # One API call for Atrial Fibrillation

        res = {'Predicted HR': final_pr, 'RR peak intervals': t_diff_afib,
               'A Fib': afib_in, 'Tachycardia': tachy_in, 'Bradycardia': brady_in}
        # return ppg_sig, hr_extracted, final_pr, afib_in, tachy_in, brady_in, data_valid
        return res, ppg_bpf, t_diff_afib,peaks_all2,final_pr
    else:
        statement = 'Data missing for over 2 minutes , PPG analysis not done'
        return statement


# save ppg_sig, hr_extracted, final_pr, afib, tachy, brady, data_valid

data = pd.read_csv(r'C:\Users\Yuvraj\Desktop\HW\HeartWatch\heart_rate\PPG\PPG_data_new-2021-10-28.csv')
result,ppg_bpf,rr_int, peaks_all2 ,final_pr= ailments_stats(data.iloc[0:30,2].to_list())
# print(result)
# print(ppg_bpf)
# print(rr_int)
# print(final_pr)
# plt.title('rr intervals: {}'.format(rr_int*1000))
# plt.plot(ppg_bpf)
# plt.text(0.,0.9,rr_int,fontsize=10)
# plt.scatter(peaks_all2,ppg_bpf[peaks_all2])
# plt.show()
# print(result)