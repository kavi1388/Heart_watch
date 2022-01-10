import pandas as pd
import numpy as np
from scipy import signal
import statistics as sts
from scipy.stats import kurtosis, skew
from scipy.stats import iqr
from scipy.signal import butter, lfilter
import math
from keras.models import load_model
import datetime
from bitstring import BitArray
import scipy.signal as signal
from scipy.signal import find_peaks
# import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


########### Feature Extraction ########


def sma(x):
    sma_val = 0
    for i in x:
        sma_val += abs(i)
    return sma_val


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size)


def peak_to_peak(s):
    max_val = max(s)
    min_val = min(s)
    return max_val - min_val


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


def scale(X):
    return (X - np.mean(X)) / np.std(X)


def low_pass_IIR(data, fl, samp_f, order):
    b, a = signal.butter(order, fl / (samp_f / 2), btype='low', output='ba')
    low_data = signal.lfilter(b, a, data)
    return low_data


def filter(acc):
    fs = 20
    a_x = acc['ACC_X (in g)']
    a_gx = butter_bandpass_filter(a_x, 0.19, 5, fs, 3)
    a_gx = normalize(a_gx)
    a_gx = a_gx - np.mean(a_gx)

    a_y = acc['ACC_Y (in g)']
    a_gy = butter_bandpass_filter(a_y, 0.19, 5, fs, 3)
    a_gy = normalize(a_gy)
    a_gy = a_gy - np.mean(a_gy)

    a_z = acc['ACC_Z (in g)']
    a_gz = butter_bandpass_filter(a_z, 0.19, 5, fs, 3)
    a_gz = normalize(a_gz)
    a_gz = a_gz - np.mean(a_gz)

    return a_gx, a_gy, a_gz


def svm(x, y, z):
    svm_val = 0
    x = x.tolist()
    y = y.tolist()
    z = z.tolist()
    for i in range(0, len(x)):
        svm_val += math.sqrt(abs(x[i]) ** 2 + abs(y[i]) ** 2 + abs(z[i]) * 2)
    return svm_val


def corr(a, b, c):
    non_corr_count = 0
    corr1 = np.corrcoef(a, b)
    corr2 = np.corrcoef(a, c)
    corr3 = np.corrcoef(b, c)
    if math.isnan(corr1[0, 1]):
        non_corr_count += 1
    if math.isnan(corr2[0, 1]):
        non_corr_count += 1
    if math.isnan(corr3[0, 1]):
        non_corr_count += 1

    return non_corr_count


def extract_features(data, window_size=20):
    col_names = ['max_x', 'min_x', 'mean_x', 'std_x', 'median_x', 'skew_x', 'kurtosis_x', 'iqr_x', 'sma_x', 'p2p_x', \
                 'max_ux', 'min_ux', 'mean_ux', 'std_ux', 'median_ux', \
                 'max_y', 'min_y', 'mean_y', 'std_y', 'median_y', 'skew_y', 'kurtosis_y', 'iqr_y', 'sma_y', 'p2p_y', \
                 'max_uy', 'min_uy', 'mean_uy', 'std_uy', 'median_uy',
                 'max_z', 'min_z', 'mean_z', 'std_z', 'median_z', 'skew_z', 'kurtosis_z', 'iqr_z', 'sma_z', 'p2p_z', \
                 'max_uz', 'min_uz', 'mean_uz', 'std_uz', 'median_uz', 'svm', 'nan_corr_count']
    extractFeature_df = pd.DataFrame(columns=col_names)

    for (start, end) in windows(data['timestep'], window_size):
        extract_list = []
        labels = np.empty((0))
        a_ux, a_uy, a_uz = filter(data[start:end])

        x = data["ACC_X (in g)"][start:end]
        max_x, min_x, mean_x, std_x, median_x = max(x), min(x), np.mean(x), np.std(x), sts.median(x)
        skew_x, kurtosis_x, iqr_x = skew(x), kurtosis(x), iqr(x)
        sma_x = sma(x)
        p2p_x = peak_to_peak(x)

        max_ux, min_ux, mean_ux, std_ux, median_ux = max(a_ux), min(a_ux), np.mean(a_ux), np.std(a_ux), sts.median(a_ux)
        extract_list.extend((max_x, min_x, mean_x, std_x, median_x, skew_x, kurtosis_x, iqr_x, sma_x, p2p_x, max_ux,
                             min_ux, mean_ux, std_ux, median_ux))

        y = data["ACC_Y (in g)"][start:end]
        max_y, min_y, mean_y, std_y, median_y = max(y), min(y), np.mean(y), np.std(y), sts.median(y)
        skew_y, kurtosis_y, iqr_y = skew(y), kurtosis(y), iqr(y)
        sma_y = sma(y)
        p2p_y = peak_to_peak(y)
        max_uy, min_uy, mean_uy, std_uy, median_uy = max(a_uy), min(a_uy), np.mean(a_uy), np.std(a_uy), sts.median(a_uy)
        extract_list.extend((max_y, min_y, mean_y, std_y, median_y, skew_y, kurtosis_y, iqr_y, sma_y, p2p_y, max_uy,
                             min_uy, mean_uy, std_uy, median_uy))

        z = data["ACC_Z (in g)"][start:end]
        max_z, min_z, mean_z, std_z, median_z = max(z), min(z), np.mean(z), np.std(z), sts.median(z)
        skew_z, kurtosis_z, iqr_z = skew(z), kurtosis(z), iqr(z)
        sma_z = sma(z)
        p2p_z = peak_to_peak(z)
        max_uz, min_uz, mean_uz, std_uz, median_uz = max(a_uz), min(a_uz), np.mean(a_uz), np.std(a_uz), sts.median(a_uz)
        extract_list.extend((max_z, min_z, mean_z, std_z, median_z, skew_z, kurtosis_z, iqr_z, sma_z, p2p_z, max_uz,
                             min_uz, mean_uz, std_uz, median_uz))
        svm_val = svm(x, y, z)
        nan_corr_count = corr(x, y, z)
        extract_list.extend((svm_val, nan_corr_count))

        df_length = len(extractFeature_df)
        extractFeature_df.loc[df_length] = extract_list

    return extractFeature_df


####### Step Counter #####

def DecimalToBinary(num):
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


################# step counter ###############

def to_decimal_step_counter(acc):
    acc_Xind = []
    # for j in range(acc.shape[0]):
    for i in range(1, 121, 6):
        acc_Xind.append(DecimalToBinary(acc.iloc[i + 1])[-1::-1] + DecimalToBinary(acc.iloc[i])[-1::-1])

    acc_X = []
    for i in range(len(acc_Xind)):
        acc_X.append(as_signed_big(acc_Xind[i]))
    acc_X = np.asarray(acc_X)

    acc_Yind = []
    # for j in range(acc.shape[0]):
    for i in range(3, 121, 6):
        acc_Yind.append(DecimalToBinary(acc.iloc[i + 1])[-1::-1] + DecimalToBinary(acc.iloc[i])[-1::-1])

    acc_Y = []
    for i in range(len(acc_Yind)):
        acc_Y.append(as_signed_big(acc_Yind[i]))
    acc_Y = np.asarray(acc_Y)

    acc_Zind = []
    # for j in range(acc.shape[0]):
    for i in range(5, 121, 6):
        acc_Zind.append(DecimalToBinary(acc.iloc[i + 1])[-1::-1] + DecimalToBinary(acc.iloc[i])[-1::-1])

    acc_Z = []
    for i in range(len(acc_Zind)):
        acc_Z.append(as_signed_big(acc_Zind[i]))
    acc_Z = np.asarray(acc_Z)

    for index_z in range(19, len(acc_Z), 20):
        acc_Z[index_z] = np.mean(acc_Z[index_z - 19: index_z - 1])

    return acc_X, acc_Y, acc_Z


def step_count(acc_X, acc_Y, acc_Z):
    a_x = acc_X * 18.3 / 128.0 / 1000.0 + 0.06
    a_y = acc_Y * 18.3 / 128.0 / 1000.0 + 0.06
    a_z = acc_Z * 18.3 / 128.0 / 1000.0 + 0.06

    x_abs = abs(a_x)
    y_abs = abs(a_y)
    z_abs = abs(a_z)
    svm = []

    for i in range(0, len(a_y)):
        svm.append(math.sqrt(x_abs[i] ** 2 + y_abs[i] * 2 + z_abs[i] ** 2))

    peaks, properties = find_peaks(svm, height=2, prominence=1)
    peaks_heights = properties['peak_heights']
    step = len(peaks)

    peaks_heights_4root = [(l * (float(1 / 4))) / 2 for l in peaks_heights]
    peaks_height_sum = np.sum(peaks_heights_4root)

    stride = float('%.2f' % ((np.mean(peaks_heights_4root) - 0.1) * 2))
    peaks_height_sum = float('%.2f' % (peaks_height_sum))

    if len(peaks) == 0 or len(peaks_heights) == 0:
        step, peaks_height_sum, stride = 0, 0, 0

    return step, peaks_height_sum, stride


############## Activity and fall ##############


def convert_to_decimal(acc):
    fs = 20
    acc['timestep'] = [d.time() for d in acc['timestep']]
    acc['timestep'] = acc['timestep'].astype(str)
    time_val = acc['timestep'].to_numpy()

    time_step_v = []
    for time in time_val:
        time_step_v.append(sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":"))))

    time_step = np.linspace(time_step_v[0], time_step_v[-1], len(time_step_v) * fs)

    acc_Xind = []
    for j in range(acc.shape[0]):
        for i in range(1, 121, 6):
            acc_Xind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_X = np.asarray(acc_Xind)

    acc_Yind = []
    for j in range(acc.shape[0]):
        for i in range(3, 121, 6):
            acc_Yind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_Y = np.asarray(acc_Yind)

    acc_Zind = []
    for j in range(acc.shape[0]):
        for i in range(5, 121, 6):
            acc_Zind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_Z = np.asarray(acc_Zind)
    acc_Z_convert = acc_Z.copy()

    for index_z in range(19, len(acc_Z_convert), 20):
        acc_Z_convert[index_z] = np.mean(acc_Z_convert[index_z - 19: index_z - 1])

    df_activity = pd.DataFrame()

    df_activity['ACC_X (in g)'] = pd.Series(acc_X)
    df_activity['ACC_Y (in g)'] = pd.Series(acc_Y)
    df_activity['ACC_Z (in g)'] = pd.Series(acc_Z)
    df_activity['timestep'] = pd.Series(time_step)

    df_fall = pd.DataFrame()

    df_fall['ACC_X (in g)'] = pd.Series(acc_X)
    df_fall['ACC_Y (in g)'] = pd.Series(acc_Y)
    df_fall['ACC_Z (in g)'] = pd.Series(acc_Z_convert)
    df_fall['timestep'] = pd.Series(time_step)

    return df_activity, df_fall


def find_slop_distance(df, col_name):
    slop_distance = pd.DataFrame(columns=['distance', 'theta', 'timestep'])

    for index, row in df.iterrows():

        first_point = row[col_name]
        first_point_moment = row['timestep']
        if index == len(df) - 1:
            break

        else:
            second_point_moment = df.loc[index + 1, 'timestep']
            second_point = df.loc[index + 1, col_name]
        distance = abs(second_point - first_point)
        theta = (math.atan2((second_point - first_point), (second_point_moment - first_point_moment)) * 180 / math.pi)
        slop_distance.loc[len(slop_distance)] = [distance, theta, second_point_moment]

    return slop_distance


def delete_outlier(window_df_decimal_fall):
    index_row = window_df_decimal_fall[
        (window_df_decimal_fall['ACC_X (in g)'] >= 5) | (window_df_decimal_fall['ACC_X (in g)'] <= -5)].index
    window_df_decimal_fall.drop(index_row, inplace=True)

    index_row = window_df_decimal_fall[
        (window_df_decimal_fall['ACC_Y (in g)'] >= 5) | (window_df_decimal_fall['ACC_Y (in g)'] <= -5)].index
    window_df_decimal_fall.drop(index_row, inplace=True)

    index_row = window_df_decimal_fall[
        (window_df_decimal_fall['ACC_Z (in g)'] >= 5) | (window_df_decimal_fall['ACC_Z (in g)'] <= -5)].index
    window_df_decimal_fall.drop(index_row, inplace=True)

    return window_df_decimal_fall


def fall_detect(window_df_decimal_fall, threshold_x, threshold_y, threshold_z):
    # print(window_df_decimal_fall)
    window_df_decimal_fall = window_df_decimal_fall.iloc[-40:, :].reset_index(drop=True)
    # print(window_df_decimal_fall)
    fall_event_time = 0
    window_df_decimal_fall = delete_outlier(window_df_decimal_fall)

    window_df_decimal_fall = window_df_decimal_fall.reset_index(drop=True)

    peak_x = window_df_decimal_fall['ACC_X (in g)'].max()
    peak_y = window_df_decimal_fall['ACC_Y (in g)'].max()
    peak_z = window_df_decimal_fall['ACC_Z (in g)'].max()

    if peak_x <= 0.25 and peak_y <= 0.25 and peak_z <= 0.25:
        return fall_event_time


    else:
        main_peak = max(peak_x, peak_y, peak_z)
        if main_peak == peak_x:
            peak_timestamp_upLimit = window_df_decimal_fall['ACC_X (in g)'].argmax() + 20
            peak_timestamp_downLimit = window_df_decimal_fall['ACC_X (in g)'].argmax() - 20

        elif main_peak == peak_y:
            peak_timestamp_upLimit = window_df_decimal_fall['ACC_Y (in g)'].argmax() + 20
            peak_timestamp_downLimit = window_df_decimal_fall['ACC_Y (in g)'].argmax() - 20
        else:
            peak_timestamp_upLimit = window_df_decimal_fall['ACC_Z (in g)'].argmax() + 20
            peak_timestamp_downLimit = window_df_decimal_fall['ACC_Z (in g)'].argmax() - 20

        if peak_timestamp_upLimit > len(window_df_decimal_fall):
            peak_timestamp_upLimit = len(window_df_decimal_fall)

        if peak_timestamp_downLimit < 0:
            peak_timestamp_downLimit = 0

        window_df_decimal_fall = window_df_decimal_fall.iloc[peak_timestamp_downLimit:peak_timestamp_upLimit,
                                 :].reset_index(drop=True)

        df_x = find_slop_distance(window_df_decimal_fall, 'ACC_X (in g)')
        df_y = find_slop_distance(window_df_decimal_fall, 'ACC_Y (in g)')
        df_z = find_slop_distance(window_df_decimal_fall, 'ACC_Z (in g)')

        df_x['theta'] = df_x['theta'].apply(lambda x: float('%.6f' % (x)))
        df_y['theta'] = df_y['theta'].apply(lambda x: float('%.6f' % (x)))
        df_z['theta'] = df_z['theta'].apply(lambda x: float('%.6f' % (x)))

        df_x_angle = df_x[df_x['theta'] < -75]
        df_y_angle = df_y[df_y['theta'] < -75]
        df_z_angle = df_z[df_z['theta'] < -75]

        df_x_angle = df_x_angle.drop_duplicates(subset=['distance', 'theta'], keep='last').reset_index(drop=True)
        df_y_angle = df_y_angle.drop_duplicates(subset=['distance', 'theta'], keep='last').reset_index(drop=True)
        df_z_angle = df_z_angle.drop_duplicates(subset=['distance', 'theta'], keep='last').reset_index(drop=True)

        if len(df_x_angle) == 0 or len(df_y_angle) == 0 or len(df_z_angle) == 0:
            return fall_event_time

        else:
            max_height_x_index = -1
            min_angle_x_index = -2
            max_height_y_index = -1
            min_angle_y_index = -2
            max_height_z_index = -1
            min_angle_z_index = -2

            if not (df_x_angle.empty):

                max_height_x = np.max(df_x_angle['distance'])
                min_angle_x = np.min(df_x_angle['theta'])
                max_height_x_index = df_x_angle['distance'].argmax()
                min_angle_x_index = df_x_angle['theta'].argmin()

                if min_angle_x < -90:
                    return fall_event_time

            if not (df_y_angle.empty):
                max_height_y = np.max(df_y_angle['distance'])
                min_angle_y = np.min(df_y_angle['theta'])
                max_height_y_index = df_y_angle['distance'].argmax()
                min_angle_y_index = df_y_angle['theta'].argmin()

                if min_angle_y < -90:
                    return fall_event_time

            if not (df_z_angle.empty):

                max_height_z = np.max(df_z_angle['distance'])
                min_angle_z = np.min(df_z_angle['theta'])
                max_height_z_index = df_z_angle['distance'].argmax()
                min_angle_z_index = df_z_angle['theta'].argmin()

                if min_angle_z < -90:
                    return fall_event_time

            if (max_height_x_index == min_angle_x_index) and (max_height_y_index == min_angle_y_index) \
                    and (max_height_z_index == min_angle_z_index) and (
                    max_height_x >= threshold_x or max_height_y >= threshold_y or max_height_z >= threshold_z):
                time_x = df_x_angle.loc[max_height_x_index, 'timestep']

                fall_event_time = time_x

    return fall_event_time


def main(window_df, model):
    threshold_x = 0
    threshold_y = 0
    threshold_z = 0

    window_df_step = window_df.copy()

    predict_y = []
    activity = ''
    reshaped_segments = []
    peaks_height_sum = 0
    step_count_value = 0
    stride = 0

    window_df_decimal, window_df_decimal_fall = convert_to_decimal(window_df)

    window_feature = extract_features(window_df_decimal)

    reshaped_segments = np.asarray(window_feature, dtype=np.float32).reshape(-1, 10, 47)

    predict_y = model.predict(reshaped_segments)

    window_feature = window_feature.drop(window_feature.index[range(0, len(window_feature))])
    window_feature = window_feature.dropna()

    x_groupby = window_df_decimal_fall.groupby(['ACC_X (in g)']).size().reset_index(name='counts')
    y_groupby = window_df_decimal_fall.groupby(['ACC_Y (in g)']).size().reset_index(name='counts')
    z_groupby = window_df_decimal_fall.groupby(['ACC_Z (in g)']).size().reset_index(name='counts')

    if (x_groupby['counts'] >= 180).any() and (y_groupby['counts'] >= 180).any() and (z_groupby['counts'] >= 180).any():
        result_index = 2  ### sleep activity


    else:
        result_index = predict_y.argmax()

    if result_index == 0:
        activity = 'Sit'
        threshold_x = 0.4
        threshold_y = 0.4
        threshold_z = 0.4

    if result_index == 1:
        activity = 'Walk'
        threshold_x = 0.5
        threshold_y = 0.5
        threshold_z = 0.5

        X, Y, Z = to_decimal_step_counter(window_df_step)
        step_count_value, peaks_height_sum, stride = step_count(X, Y, Z)

    if result_index == 2:
        activity = 'Sleep'
        threshold_x = 0.4
        threshold_y = 0.4
        threshold_z = 0.4

    fall_event_time = fall_detect(window_df_decimal_fall, threshold_x, threshold_y, threshold_z)

    if fall_event_time != 0:
        fall_event_time = str(datetime.timedelta(seconds=fall_event_time))

    return activity, fall_event_time, step_count_value, peaks_height_sum, stride


########### Read data #############


def call_model_(data2):
    window_df = pd.DataFrame(columns=range(0, 121))
    acc_data_df = pd.DataFrame(columns=['data', '_id', 'app_date', 'Gap'])
    CNN_model = load_model('./heart_rate/Accelerometer/CNN_walk_sit_feature20_10second_originalZ.h5')
    window_size = 10
    two_minutes = 120
    predict_activity = ''
    fall_timestamps = 0

    # data_ = json.loads(data2)
    # print('data_')
    # print(data_)
    # for data_ in data2:
    for index, dt in enumerate(data2):
        # print('index ',index)
        # print('dt ' ,dt)
        acc_data_df.loc[index, 'data'] = dt['data']
        acc_data_df.loc[index, '_id'] = dt['_id']
        acc_data_df.loc[index, 'app_date'] = dt['app_date']

    acc_data_df['app_date'] = pd.to_datetime(acc_data_df['app_date'], format='%d/%m/%Y %H:%M:%S')
    for index, row in acc_data_df.iterrows():

        one_second = row['data']
        if one_second[0] == 61 and len(one_second) == 121:

            if len(window_df) < window_size:
                window_df.loc[index, 0:121] = one_second
                window_df.loc[index, 'timestep'] = row['app_date']

            if len(window_df) == window_size:
                activity, fall, step_count_value, peaks_height_sum, stride = main(window_df, CNN_model)
                predict_activity = activity
                fall_timestamps = fall
                window_df = window_df.drop(window_df.index[range(0, len(window_df))])
                window_df = window_df.dropna()

    if fall != 0 and fall is not None:
        fall_flag = True
    else:
        fall_flag = False
    res = {"activity": predict_activity, "fall_flag": fall_flag, "fall_time": fall_timestamps,
           "steps": step_count_value, "distance": peaks_height_sum, "stride": stride}

    return res

# data = [{"app_date":"24/12/2021 00:36:50","data":[61,-94,-1,112,-1,82,-1,-88,-1,107,-1,68,-1,-93,-1,101,-1,80,-1,-89,-1,102,-1,80,-1,-90,-1,101,-1,79,-1,-89,-1,105,-1,78,-1,-93,-1,113,-1,74,-1,-93,-1,96,-1,71,-1,-87,-1,103,-1,66,-1,-89,-1,111,-1,72,-1,-92,-1,115,-1,75,-1,-99,-1,112,-1,88,-1,-96,-1,106,-1,86,-1,-90,-1,104,-1,78,-1,-98,-1,101,-1,85,-1,-94,-1,102,-1,80,-1,-95,-1,116,-1,73,-1,-93,-1,112,-1,72,-1,-95,-1,105,-1,77,-1,-101,-1,115,-1,73,28],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:50","data":[61,-60,-1,-102,-1,40,-1,-56,-1,-111,-1,43,-1,-55,-1,127,-1,53,-1,-63,-1,119,-1,83,-1,-68,-1,107,-1,75,-1,-65,-1,110,-1,66,-1,-74,-1,106,-1,100,-1,-87,-1,105,-1,87,-1,-75,-1,93,-1,70,-1,-81,-1,110,-1,66,-1,-84,-1,113,-1,61,-1,-82,-1,120,-1,57,-1,-88,-1,123,-1,64,-1,-89,-1,123,-1,60,-1,-91,-1,-126,-1,62,-1,-91,-1,118,-1,65,-1,-88,-1,123,-1,63,-1,-91,-1,117,-1,75,-1,-87,-1,121,-1,69,-1,-87,-1,122,-1,71,108],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:49","data":[61,-99,-1,-106,-1,52,-1,-99,-1,-106,-1,52,-1,-99,-1,-106,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-107,-1,50,-1,-99,-1,-107,-1,50,-1,-98,-1,-106,-1,50,-1,-99,-1,-105,-1,52,-1,-100,-1,-105,-1,51,-1,-99,-1,-105,-1,50,-1,-100,-1,-106,-1,50,-1,-107,-1,-100,-1,50,-1,-108,-1,-101,-1,49,-1,-113,-1,-101,-1,54,-1,-115,-1,-96,-1,54,-1,-101,-1,-95,-1,38,-1,-88,-1,-83,-1,22,-1,-85,-1,-83,-1,32,-1,-72,-1,-77,-1,19,-1,-64,-1,-94,-1,26,42],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:49","data":[61,-99,-1,-105,-1,49,-1,-98,-1,-104,-1,51,-1,-98,-1,-104,-1,51,-1,-99,-1,-105,-1,52,-1,-98,-1,-104,-1,51,-1,-97,-1,-104,-1,51,-1,-97,-1,-104,-1,50,-1,-98,-1,-105,-1,50,-1,-98,-1,-106,-1,50,-1,-98,-1,-106,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-106,-1,52,-1,-97,-1,-104,-1,51,-1,-98,-1,-104,-1,52,-1,-98,-1,-106,-1,51,-1,-98,-1,-107,-1,51,-1,-99,-1,-107,-1,50,-1,-98,-1,-107,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-106,-1,51,27],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:48","data":[61,-99,-1,-104,-1,51,-1,-98,-1,-103,-1,50,-1,-99,-1,-102,-1,51,-1,-98,-1,-102,-1,51,-1,-99,-1,-102,-1,53,-1,-98,-1,-103,-1,51,-1,-99,-1,-104,-1,51,-1,-98,-1,-105,-1,49,-1,-98,-1,-104,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-104,-1,50,-1,-98,-1,-104,-1,52,-1,-97,-1,-105,-1,52,-1,-98,-1,-105,-1,52,-1,-99,-1,-104,-1,50,-1,-98,-1,-103,-1,50,-1,-98,-1,-103,-1,50,-1,-98,-1,-104,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-104,-1,50,44],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:47","data":[61,-98,-1,-105,-1,50,-1,-97,-1,-105,-1,50,-1,-97,-1,-106,-1,51,-1,-97,-1,-106,-1,50,-1,-99,-1,-106,-1,51,-1,-98,-1,-105,-1,50,-1,-98,-1,-103,-1,48,-1,-99,-1,-104,-1,47,-1,-99,-1,-104,-1,47,-1,-99,-1,-104,-1,50,-1,-98,-1,-104,-1,50,-1,-99,-1,-104,-1,50,-1,-98,-1,-103,-1,50,-1,-98,-1,-103,-1,52,-1,-98,-1,-103,-1,51,-1,-98,-1,-103,-1,52,-1,-98,-1,-104,-1,51,-1,-98,-1,-104,-1,52,-1,-98,-1,-104,-1,50,-1,-98,-1,-103,-1,49,30],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:47","data":[61,-96,-1,-106,-1,50,-1,-96,-1,-106,-1,52,-1,-96,-1,-105,-1,51,-1,-96,-1,-107,-1,51,-1,-96,-1,-106,-1,50,-1,-96,-1,-105,-1,50,-1,-97,-1,-104,-1,49,-1,-98,-1,-103,-1,49,-1,-96,-1,-103,-1,51,-1,-96,-1,-103,-1,52,-1,-97,-1,-104,-1,50,-1,-96,-1,-107,-1,48,-1,-96,-1,-107,-1,48,-1,-95,-1,-106,-1,51,-1,-96,-1,-106,-1,49,-1,-97,-1,-106,-1,51,-1,-98,-1,-106,-1,50,-1,-98,-1,-104,-1,49,-1,-98,-1,-104,-1,51,-1,-99,-1,-104,-1,51,42],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:46","data":[61,-98,-1,-104,-1,51,-1,-98,-1,-105,-1,50,-1,-97,-1,-106,-1,52,-1,-97,-1,-107,-1,52,-1,-98,-1,-106,-1,51,-1,-97,-1,-106,-1,51,-1,-97,-1,-105,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-105,-1,49,-1,-99,-1,-105,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,52,-1,-99,-1,-106,-1,51,-1,-100,-1,-107,-1,50,-1,-100,-1,-107,-1,50,-1,-98,-1,-106,-1,50,-1,-98,-1,-106,-1,48,-1,-96,-1,-106,-1,49,8],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:45","data":[61,-98,-1,-104,-1,51,-1,-98,-1,-105,-1,50,-1,-97,-1,-106,-1,52,-1,-97,-1,-107,-1,52,-1,-98,-1,-106,-1,51,-1,-97,-1,-106,-1,51,-1,-97,-1,-105,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-105,-1,49,-1,-99,-1,-105,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,52,-1,-99,-1,-106,-1,51,-1,-100,-1,-107,-1,50,-1,-100,-1,-107,-1,50,-1,-98,-1,-106,-1,50,-1,-98,-1,-106,-1,48,-1,-96,-1,-106,-1,49,8],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:44","data":[61,-98,-1,-104,-1,51,-1,-98,-1,-105,-1,50,-1,-97,-1,-106,-1,52,-1,-97,-1,-107,-1,52,-1,-98,-1,-106,-1,51,-1,-97,-1,-106,-1,51,-1,-97,-1,-105,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-105,-1,49,-1,-99,-1,-105,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,52,-1,-99,-1,-106,-1,51,-1,-100,-1,-107,-1,50,-1,-100,-1,-107,-1,50,-1,-98,-1,-106,-1,50,-1,-98,-1,-106,-1,48,-1,-96,-1,-106,-1,49,8],"_id":"6052e4dc605f500004ef6d3f"},{"app_date":"24/12/2021 00:36:43","data":[61,-98,-1,-104,-1,51,-1,-98,-1,-105,-1,50,-1,-97,-1,-106,-1,52,-1,-97,-1,-107,-1,52,-1,-98,-1,-106,-1,51,-1,-97,-1,-106,-1,51,-1,-97,-1,-105,-1,51,-1,-98,-1,-105,-1,51,-1,-98,-1,-105,-1,49,-1,-99,-1,-105,-1,49,-1,-98,-1,-105,-1,49,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,50,-1,-98,-1,-105,-1,52,-1,-99,-1,-106,-1,51,-1,-100,-1,-107,-1,50,-1,-100,-1,-107,-1,50,-1,-98,-1,-106,-1,50,-1,-98,-1,-106,-1,48,-1,-96,-1,-106,-1,49,8],"_id":"6052e4dc605f500004ef6d3f"}]

# result=call_model_(data)
# print(result)