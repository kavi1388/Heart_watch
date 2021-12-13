from ppg_hr import *
from custom_modules import *
import datetime
import time
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def ailments_stats(self, ppg_list):
    strike = 0
    strike_tachy = 0
    count = 15
    count_afib = 15
    brady_in = False
    tachy_in = False
    afib_in = False
    data_valid = True
    api_type = None
    time_val = []
    ppg_bytes = []

    # reading of the input file starts here
    for d in ppg_list:
        ppg_data = d
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
        final_pr, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted, non_uniform, spo2_pred = ppg_plot_hr(
            ppg_sig, time_val, fl=1, fh=5, o=4, n=5, diff_max=10, r=5)
        resp_rate = rr_calulation(ppg_sig)

        for i in range(len(hr_extracted)):
            if 60 > hr_extracted[i] >= 40:
                strike += 1
                if strike == count:
                    brady_in = True
                    api_type = 1
            else:
                strike = 0
                brady_in = False

            if hr_extracted[i] > 100:
                strike_tachy += 1
                if strike_tachy == count:
                    tachy_in = True
                    api_type = 2
            else:
                strike_tachy = 0
                tachy_in = False
            # One API call for Bradycardia (type 1==True)

            # return 'No Bradycardia'

            # One API call for Tachycardia (type 2==True)
        if non_uniform == count_afib:
            afib_in = True
            api_type = 5

        res = {'time_interval': (time_val[0], time_val[-1]), 'predicted_HR': hr_extracted, 'predicted_SPO2': spo2_pred,
               'resp_rate': resp_rate, 'rr_peak_intervals': t_diff_afib, 'a_Fib': afib_in, 'tachycardia': tachy_in,
               'bradycardia': brady_in, 'dataFrame': final_pr, 'api_type': api_type}

        return res
    else:
        statement = 'Data missing for over 2 minutes , PPG analysis not done'
        return statement


# save ppg_sig, hr_extracted, final_pr, afib, tachy, brady, data_valid

data = pd.read_csv(r'C:\Users\Yuvraj\Desktop\Live Testing\PPG_data_new-2021-10-29.csv')
hr_orig=pd.read_csv(r'C:\Users\Yuvraj\Desktop\Live Testing\hr_device_29th.csv')

hr_orig = hr_orig.rename(columns={'ISTTime': 'timestamps'})
hr_orig = hr_orig.rename(columns={'heart_data_timestamp_array': 'timestamps'})

result = ailments_stats(data.iloc[:60,2].to_list())

hr_all_df = result['dataFrame']

# plt.figure()
# plt.title('Actual Vs Predicted HR')
# plt.plot(hr_all_df['heart_data_array'], label='Actual HR')
# plt.plot(hr_all_df['heart predicted by peak time diff'],label='Predicted HR')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.title('Error Plot')
# plt.plot(np.abs(hr_all_df['heart predicted by peak time diff']-hr_all_df['heart_data_array']),label='Error in Prediction')
# plt.legend()
# plt.show()


