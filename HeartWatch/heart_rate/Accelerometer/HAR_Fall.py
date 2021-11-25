
from keras.backend import print_tensor
from numpy.lib.function_base import piecewise
import pandas as pd
import os
from keras.models import load_model
from datetime import datetime
from ..Accelerometer.save_model import main
import json
import ast

def call_model(data):
    window_df = pd.DataFrame(columns=range(0, 121))
    acc_data_df = pd.DataFrame(columns=['data', '_id', 'app_date', 'Gap'])
    CNN_model = load_model('/home/heartwatch/rajnishh/Heart_watch/HeartWatch/heart_rate/Accelerometer/CNN_walk_sit_feature20_30second_originalZ.h5')
    window_size = 30
    two_minutes = 120
    predict_activity = []
    fall_timestamps = []
    time_reading=[]
    for index, dt in enumerate(data):
        dt = ast.literal_eval(dt)

        acc_data_df.loc[index, 'data'] = dt['data']
        acc_data_df.loc[index, '_id'] = dt['_id']
        acc_data_df.loc[index, 'app_date'] = dt['app_date']
        time_reading.append(acc_data_df.loc[index , 'app_date'].split()[1])
        # print(len(dt['data']))
        # print(acc_data_df)

    ### drop time duplication ####
    # acc_data_df = acc_data_df.drop_duplicates(subset="app_date")

    #### reverse data ####
    acc_data_df = acc_data_df.iloc[::-1]

    #### check timestamps #####
    acc_data_df['app_date'] = pd.to_datetime(acc_data_df['app_date'], format='%d/%m/%Y %H:%M:%S')
    acc_data_df['Gap'] = abs((acc_data_df['app_date'].shift(-1) - acc_data_df['app_date']).dt.total_seconds())
    # print(acc_data_df)
    if (acc_data_df['Gap'] >= two_minutes).any():
        return ('There is a time gap more than 2 minutes ...'), ('Drop...')  # None, None

    else:
        for index, row in acc_data_df.iterrows():

            one_second = row['data']
            if one_second[0] == 61 and len(one_second) == 121:

                if len(window_df) < window_size:
                    window_df.loc[index, 0:121] = one_second
                    window_df.loc[index, 'timestep'] = row['app_date']

                if len(window_df) == window_size:
                    # print(window_df)
                    activity, fall = main(window_df, CNN_model)
                    predict_activity.append(activity)
                    fall_timestamps.append(fall)
                    window_df = window_df.drop(window_df.index[range(0, len(window_df))])
                    window_df = window_df.dropna()
                    # print(window_df)

    return time_reading[0], time_reading[-1], predict_activity, fall_timestamps


