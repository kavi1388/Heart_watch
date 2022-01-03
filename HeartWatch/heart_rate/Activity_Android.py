import pandas as pd
import numpy as np
from scipy import signal
import statistics as sts
from scipy.stats import kurtosis, skew
from scipy.stats import iqr
from scipy.signal import butter, lfilter
import math
from keras.models import load_model
import json
import datetime
from bitstring import BitArray
import scipy.signal as signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")
import gc
gc.collect()

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
    for j in range(acc.shape[0]):
        for i in range(1, 121, 6):
            acc_Xind.append(DecimalToBinary(acc.iloc[j, i + 1])[-1::-1] + DecimalToBinary(acc.iloc[j, i])[-1::-1])

    acc_X = []
    for i in range(len(acc_Xind)):
        acc_X.append(as_signed_big(acc_Xind[i]))
    acc_X = np.asarray(acc_X)

    acc_Yind = []
    for j in range(acc.shape[0]):
        for i in range(3, 121, 6):
            acc_Yind.append(DecimalToBinary(acc.iloc[j, i + 1])[-1::-1] + DecimalToBinary(acc.iloc[j, i])[-1::-1])

    acc_Y = []
    for i in range(len(acc_Yind)):
        acc_Y.append(as_signed_big(acc_Yind[i]))
    acc_Y = np.asarray(acc_Y)

    acc_Zind = []
    for j in range(acc.shape[0]):
        for i in range(5, 121, 6):
            acc_Zind.append(DecimalToBinary(acc.iloc[j, i + 1])[-1::-1] + DecimalToBinary(acc.iloc[j, i])[-1::-1])

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


def calc_xy_angles(x, y, z):
    x_val = x
    y_val = y
    z_val = z

    x2 = x_val * x_val
    y2 = y_val * y_val
    z2 = z_val * z_val

    result = np.sqrt(y2 + z2)
    result = x_val / result
    accel_angle_x = np.arctan(result)

    result = np.sqrt(x2 + z2)
    result = y_val / result
    accel_angle_y = np.arctan(result)

    return accel_angle_x, accel_angle_y


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

    df2 = pd.DataFrame()

    df2['ACC_X (in g)'] = pd.Series(acc_X)
    df2['ACC_Y (in g)'] = pd.Series(acc_Y)
    df2['ACC_Z (in g)'] = pd.Series(acc_Z)
    df2['timestep'] = pd.Series(time_step)

    return df2


def convert_to_decimal_Fall(acc):
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

    df2 = pd.DataFrame()

    for index_z in range(19, len(acc_Z), 20):
        acc_Z[index_z] = np.mean(acc_Z[index_z - 19: index_z - 1])

    df2 = pd.DataFrame()

    df2['ACC_X (in g)'] = pd.Series(acc_X)
    df2['ACC_Y (in g)'] = pd.Series(acc_Y)
    df2['ACC_Z (in g)'] = pd.Series(acc_Z)
    df2['timestep'] = pd.Series(time_step)

    return df2


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
    window_df_decimal_fall = delete_outlier(window_df_decimal_fall)
    window_df_decimal_fall = window_df_decimal_fall.reset_index()
    fall_event_time = 0

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
            # time_y = df_y_angle.loc[max_height_y_index , 'timestep']
            # time_z = df_z_angle.loc[max_height_z_index , 'timestep']

            fall_event_time = time_x

    return fall_event_time


def main(window_df, model):
    threshold_x = 0
    threshold_y = 0
    threshold_z = 0

    window_df_fall = window_df.copy()
    window_df_step = window_df.copy()

    predict_y = []
    activity = ''
    reshaped_segments = []
    peaks_height_sum = 0
    step_count_value = 0
    stride = 0

    window_df_decimal = convert_to_decimal(window_df)

    window_df_decimal_fall = convert_to_decimal_Fall(window_df_fall)

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

    del acc_data_df
    del window_df
    gc.collect()

    return res

# data = '[{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:12","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},' \
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},' \
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0},'\
#         '{"_id":"6052e4dc605f500004ef6d3f","app_date":"11/12/2021 18:42:13","data": [61,69,-1,41,0,-89,0,71,-1,40,0,-88,0,69,-1,38,0,-86,0,70,-1,35,0,-87,0,72,-1,35,0,-87,0,72,-1,34,0,-88,0,71,-1,35,0,-88,0,71,-1,38,0,-87,0,70,-1,39,0,-87,0,70,-1,38,0,-86,0,69,-1,36,0,-87,0,70,-1,37,0,-88,0,68,-1,40,0,-89,0,68,-1,33,0,-87,0,65,-1,38,0,-91,0,64,-1,39,0,-88,0,66,-1,29,0,-88,0,66,-1,9,0,-92,0,71,-1,-4,-1,-98,0,70,-1,1,0,-102,14],"id":0}]'
#
#
#
# result=call_model(data)
# print(result)