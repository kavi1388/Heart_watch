import pandas as pd
import numpy as np
import math
import datetime


def convert_to_decimal(acc):
    fs = 20

    # acc=acc_orig.drop_duplicates(subset="timestamps")

    acc['timestep'] = [d.time() for d in acc['timestep']]
    acc['timestep'] = acc['timestep'].astype(str)
    time_val = acc['timestep'].to_numpy()

    time_step_v = []
    for time in time_val:
        time_step_v.append(sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":"))))

    time_step = np.linspace(time_step_v[0], time_step_v[-1], len(time_step_v) * fs)

    acc_Xind = []
    for j in range(acc.shape[0]):
        for i in range(0, 120, 6):
            acc_Xind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_X = np.asarray(acc_Xind)

    acc_Yind = []
    for j in range(acc.shape[0]):
        for i in range(2, 120, 6):
            acc_Yind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_Y = np.asarray(acc_Yind)

    acc_Zind = []
    for j in range(acc.shape[0]):
        for i in range(4, 120, 6):
            acc_Zind.append((int(hex(np.int64(acc.iloc[j, i + 1]).item()) +
                                 hex(np.int64(acc.iloc[j, i]).item()).split('x')[-1], 16) >> 4) / 128.0)
    acc_Z = np.asarray(acc_Zind)

    df2 = pd.DataFrame()

    df2['ACC_X (in g)'] = pd.Series(acc_X)
    df2['ACC_Y (in g)'] = pd.Series(acc_Y)
    df2['ACC_Z (in g)'] = pd.Series(acc_Z)
    df2['timestep'] = pd.Series(time_step)

    return df2


def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size)


def segment_signal(data, window_size=200):
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    for (start, end) in windows(data['ACC_X (in g)'], window_size):
        x = data["ACC_X (in g)"][start:end]
        y = data["ACC_Y (in g)"][start:end]
        z = data["ACC_Z (in g)"][start:end]
        if (len(data['ACC_X (in g)'][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])

    return segments


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


# def detect_outlier(data):
#     print(data)
#     outliers=[]
#     indices = []
#     threshold= 4
#     mean = np.mean(data.iloc[: ,1 ].values)
#     std = np.std(data.iloc[: , 1].values)

#     for index , row in data.iterrows():

#         z_score= (row['svm'] - mean)/std
#         if np.abs(z_score) > threshold:
#             outliers.append(row['svm'])
#             indices.append(row['timestep'])
#     return outliers , indices


def fall_detect(acc_data_df):
    fall_event_time = [0]

    # df = convert_to_decimal(acc_data_df)

    df_x = find_slop_distance(acc_data_df, 'ACC_X (in g)')
    df_y = find_slop_distance(acc_data_df, 'ACC_Y (in g)')

    df_x_angle = df_x[df_x['theta'] < -75]
    df_y_angle = df_y[df_y['theta'] < -75]

    if len(df_x_angle > 0):
        for t in df_x_angle['timestep']:
            if t - fall_event_time[-1] > 20:
                fall_event_time.append(t)

    if len(df_y_angle > 0):
        for t in df_y_angle['timestep']:
            if t - fall_event_time[-1] > 20:
                fall_event_time.append(t)

    # df_x_rfp = remove_false_positive(df_x)
    # df_y_rfp = remove_false_positive(df_y)
    # mean_distance_x = np.mean(df_x['distance'].values)
    # mean_distance_y = np.mean(df_y['distance'].values)
    # df_x = df_x[df_x['distance'] > mean_distance_x ]
    # df_y = df_y[df_y['distance'] > mean_distance_y ]
    # df_x_rfp = remove_false_positive(df_x)
    # df_y_rfp = remove_false_positive(df_y)

    return fall_event_time


def main(window_df, model):
    window_df_decimal = convert_to_decimal(window_df)
    segment = segment_signal(window_df_decimal)
    predict_y = model.predict(segment)
    onehot_predict = []
    fall_event_time = []
    for elem in predict_y:
        max_val = max(elem)
        elem = [1 if i == max_val else 0 for i in elem]
        onehot_predict.append(elem)

    result_index = onehot_predict[0].index(1)
    if result_index == 0:
        activity = 'Walk'

    elif result_index == 1:
        activity = 'Sit'

    elif result_index == 2:
        activity = 'run'

    elif result_index == 3:
        activity = 'Up_Stair'

    elif result_index == 4:
        activity = 'Down_Stair'

    elif result_index == 5:
        activity = 'LayDown'

    fall_event_time = fall_detect(window_df_decimal)
    if len(fall_event_time) == 1:
        fall_event_time[0] = ('No Fall')
    else:
        for i in range(len(fall_event_time)):
            fall_event_time[i] = str(datetime.timedelta(seconds=fall_event_time[i]))
        fall_event_time = fall_event_time[1:]

        # print(activity)
    # print(fall_event_time)  
    # print(window_df_decimal['timestep'])
    return activity, fall_event_time


if __name__ == '__main__':
    main(window_df, model)
