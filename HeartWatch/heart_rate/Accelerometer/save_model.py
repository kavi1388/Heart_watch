
import pandas as pd
import numpy as np
import math
import datetime
from ..Accelerometer.FeatureExtractionFunction import extract_features


def calc_xy_angles(x, y, z):
    x_val = x  # -np.mean(x)
    y_val = y  # -np.mean(y)
    z_val = z  # -np.mean(z)

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


def low_pass_IIR(data, fl, samp_f, order):
    b, a = signal.butter(order, fl / (samp_f / 2), btype='low', output='ba')
    low_data = signal.lfilter(b, a, data)
    return low_data


def high_pass_IIR(data, fh, samp_f, order):
    b, a = signal.butter(order, fh / (samp_f / 2), btype='high', output='ba')
    high_data = signal.lfilter(b, a, data)
    return high_data


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
    slop_distance = pd.DataFrame(columns=['distance', 'slop', 'theta', 'timestep'])

    for index, row in df.iterrows():

        first_point = row[col_name]
        first_point_moment = row['timestep']
        if index == len(df) - 1:
            break

        else:
            second_point_moment = df.loc[index + 1, 'timestep']
            second_point = df.loc[index + 1, col_name]
        distance = abs(second_point - first_point)
        slop = 0  # (second_point - first_point) / (second_point_moment - first_point_moment)
        theta = (math.atan2((second_point - first_point), (second_point_moment - first_point_moment)) * 180 / math.pi)
        slop_distance.loc[len(slop_distance)] = [distance, slop, theta, second_point_moment]

    return slop_distance


def fall_detect(window_to_decimal, threshold_x, threshold_y, threshold_z=0.5):
    fall_flag = False
    fall_event_time = [0]
    df_x = find_slop_distance(window_to_decimal, 'ACC_X (in g)')
    df_y = find_slop_distance(window_to_decimal, 'ACC_Y (in g)')
    df_z = find_slop_distance(window_to_decimal, 'ACC_Z (in g)')

    df_x_angle = df_x[df_x['theta'] < -75]
    df_y_angle = df_y[df_y['theta'] < -75]
    df_z_angle = df_z[df_z['theta'] < -75]

    df_x_angle = df_x_angle.reset_index()
    df_y_angle = df_y_angle.reset_index()
    df_z_angle = df_z_angle.reset_index()

    if len(df_x_angle) == 0 or len(df_y_angle) == 0:
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
            max_height_x_count = df_x_angle[df_x_angle['distance'] == max_height_x]
            min_angle_x_count = df_x_angle[df_x_angle['theta'] == min_angle_x]

            if len(max_height_x_count) >= 2 or len(min_angle_x_count) >= 2 or min_angle_x < -90:
                return fall_event_time

        if not (df_y_angle.empty):
            max_height_y = np.max(df_y_angle['distance'])
            min_angle_y = np.min(df_y_angle['theta'])
            max_height_y_index = df_y_angle['distance'].argmax()
            min_angle_y_index = df_y_angle['theta'].argmin()
            max_height_y_count = df_y_angle[df_y_angle['distance'] == max_height_y]
            min_angle_y_count = df_y_angle[df_y_angle['theta'] == min_angle_y]

            if len(max_height_y_count) >= 2 or len(min_angle_y_count) >= 2 or min_angle_y < -90:
                return fall_event_time

        if not (df_z_angle.empty):

            max_height_z = np.max(df_z_angle['distance'])
            min_angle_z = np.min(df_z_angle['theta'])
            max_height_z_index = df_z_angle['distance'].argmax()
            min_angle_z_index = df_z_angle['theta'].argmin()
            max_height_z_count = df_z_angle[df_z_angle['distance'] == max_height_z]
            min_angle_z_count = df_z_angle[df_z_angle['theta'] == min_angle_z]

            if len(max_height_z_count) >= 2 or len(min_angle_z_count) >= 2 or min_angle_z < -90:
                return fall_event_time

        if (max_height_x_index == min_angle_x_index) and (max_height_y_index == min_angle_y_index) \
                and (max_height_z_index == min_angle_z_index) and (
                max_height_x >= threshold_x or max_height_y >= threshold_y or max_height_z >= threshold_z):
            time_x = df_x_angle.loc[max_height_x_index, 'timestep']
            time_y = df_y_angle.loc[max_height_y_index, 'timestep']

            if abs(time_x - time_y) < 20:
                fall_flag = True
                fall_event_time[0] = time_x

            else:
                fall_Flag = True
                fall_event_time[0] = time_x
                fall_event_time[0] = time_y

    return fall_event_time


def main(window_df, model):
    threshold_x = 0
    threshold_y = 0
    threshold_z = 0

    window_df_fall = window_df.copy()

    predict_y = []
    activity = ''
    # step_count_value = [0]
    reshaped_segments = []
    # step_count_value = []
    window_df_decimal = convert_to_decimal(window_df)
    window_df_decimal_fall = convert_to_decimal_Fall(window_df_fall)
    # X , Y , Z  = sc.to_decimal(window_df)
    # step_count_value , distance = sc.step_count(X,Y,Z)

    window_feature = extract_features(window_df_decimal)

    reshaped_segments = np.asarray(window_feature, dtype=np.float32).reshape(-1, 30, 47)

    predict_y = model.predict(reshaped_segments)

    window_feature = window_feature.drop(window_feature.index[range(0, len(window_feature))])
    window_feature = window_feature.dropna()
    fall_event_time = []

    result_index = predict_y.argmax()

    if result_index == 0:
        activity = 'Sit'
        threshold_x = 0.5
        threshold_y = 0.5
        threshold_z = 0.5

    if result_index == 1:
        activity = 'Walk'
        threshold_x = 0.5
        threshold_y = 0.5
        threshold_z = 0.5

    fall_event_time = fall_detect(window_df_decimal_fall, threshold_x, threshold_y, threshold_z)
    if fall_event_time[0] == 0:
        fall_event_time[0] = 'No Fall'
    else:
        for i in range(len(fall_event_time)):
            fall_event_time[i] = str(datetime.timedelta(seconds=fall_event_time[i]))

    return activity, fall_event_time


if __name__ == '__main__':
    main(window_df, model)


