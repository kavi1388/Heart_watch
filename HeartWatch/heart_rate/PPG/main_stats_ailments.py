
from ppg_hr import *
from custom_modules import *
import datetime
import time
import json


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
        a = ppg_list[ind].replace('\"', '')
        json_acceptable_string = a.replace('\\', '').replace("'", '"')
        ppg_data = json.loads(json_acceptable_string)
        ppg_sec = ppg_data['heart_rate_voltage']['data']
        time_val.append(ppg_data['heart_rate_voltage']['app_date'].split()[1])
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
        final_pr, ppg_21, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted = ppg_plot_hr(ppg_sig, time_val)

        for i in range(len(hr_extracted)):
            if 60 > hr_extracted[i] >= 40:
                strike += 1
            else:
                strike = 0

            if strike == count:
                print('Patient has Sinus Bradycardia')
                brady_in = True

                # One API call for Bradycardia

            if 100 < hr_extracted[i] <= 130:
                strike_tachy += 1
            else:
                strike_tachy = 0

            if strike_tachy == count:
                print('Patient has Sinus Tachycardia')
                tachy_in = True

                # One API call for Tachycardia

        for i in range(len(t_diff_afib) - 1):
            if t_diff_afib[i + 1] - t_diff_afib[i] > 10:
                strike_afib += 1
            else:
                strike_afib = 0
            if strike_afib == count_afib:
                print('Patient has Atrial Fibrillation')
                afib_in = True

        # One API call for Atrial Fibrillation
        return ppg_sig, hr_extracted, final_pr, afib_in, tachy_in, brady_in, data_valid

    else:
        statement = 'Data missing for over 2 minutes , PPG analysis not done'
        return ppg_sig, data_valid


# save afib,tachy,brady,ppg_sig,hr_extracted,final_pr,statement

print('Main')
# result = ailments_stats([
#    "\"{\\'heart_rate_voltage\\': {\\'data\\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \\'_id\\': \\'6052e4dc605f500004ef6d3f\\', \\'app_date\\': \\'29/09/2021 16:35:00\\'}}\"",
#    "\"{\\'heart_rate_voltage\\': {\\'data\\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \\'_id\\': \\'6052e4dc605f500004ef6d3f\\', \\'app_date\\': \\'29/09/2021 16:35:00\\'}}\"",
#    "\"{\\'heart_rate_voltage\\': {\\'data\\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \\'_id\\': \\'6052e4dc605f500004ef6d3f\\', \\'app_date\\': \\'29/09/2021 16:35:00\\'}}\""
# ])

result = ailments_stats(['"{\'heart_rate_voltage\': {\'data\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \'_id\': \'6052e4dc605f500004ef6d3f\', \'app_date\': \'29/09/2021 16:35:00\'}}"', '"{\'heart_rate_voltage\': {\'data\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \'_id\': \'6052e4dc605f500004ef6d3f\', \'app_date\': \'29/09/2021 16:35:00\'}}"', '"{\'heart_rate_voltage\': {\'data\': [-7, 0, -1, 78, 1, -5, 78, 2, -8, 78, 3, -4, 78, 4, 5, 79, 5, 19, 79, 6, 25, 79, 7, 34, 79, 8, 44, 79, 9, 62, 79, 10, 86, 79, 11, 119, 79, 12, -116, 79, 13, -117, 79, 14, 107, 79, 15, 64, 79, 16, 40, 79, 17, 37, 79, 18, 44, 79, 19, 59, 79, 20, 67, 79, 21, 58, 79, 22, 50, 79, 23, 44, 79, 24, 48, -126], \'_id\': \'6052e4dc605f500004ef6d3f\', \'app_date\': \'29/09/2021 16:35:00\'}}"']
)
if result[-1]:
    ppg_sig, hr_extracted, final_pr, afib, tachy, brady, data_valid = result
    print(final_pr)
    print("*" * 50)
    print(ppg_sig)
    print("*" * 50)
    print(hr_extracted)
    print("*" * 50)
    print(final_pr)
    print("*" * 50)
    print(afib)
    print("*" * 50)
    print(tachy)
    print("*" * 50)
    print(brady)
    print("*" * 50)
    print(data_valid)
else:
    ppg_sig, data_valid = result
    print(ppg_sig)
    print("*" * 50)
    print(data_valid)
