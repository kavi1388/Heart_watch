
import pandas as pd
import numpy as np
import math
import datetime
from FeatureExtractionFunction import extract_features 

def convert_to_decimal (acc):

    fs = 20

    # acc=acc_orig.drop_duplicates(subset="timestamps")

    acc['timestep'] = [d.time() for d in acc['timestep']]
    acc['timestep'] = acc['timestep'].astype(str)
    time_val=acc['timestep'].to_numpy()
        
    time_step_v=[]
    for time in time_val:
        time_step_v.append(sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))) 

  
    time_step=np.linspace(time_step_v[0],time_step_v[-1],len(time_step_v)*fs)
     
    acc_Xind=[]
    for j in range(acc.shape[0]):
        for i in range(0,120,6):    
            acc_Xind.append((int(hex(np.int64(acc.iloc[j,i+1]).item())+hex(np.int64(acc.iloc[j,i]).item()).split('x')[-1],16)>>4)/128.0)
    acc_X=np.asarray(acc_Xind)

    acc_Yind=[]
    for j in range(acc.shape[0]):
        for i in range(2,120,6):    

            acc_Yind.append((int(hex(np.int64(acc.iloc[j,i+1]).item())+hex(np.int64(acc.iloc[j,i]).item()).split('x')[-1],16)>>4)/128.0)
    acc_Y=np.asarray(acc_Yind)

    acc_Zind=[]
    for j in range(acc.shape[0]):
        for i in range(4,120,6):    
            acc_Zind.append((int(hex(np.int64(acc.iloc[j,i+1]).item())+hex(np.int64(acc.iloc[j,i]).item()).split('x')[-1],16)>>4)/128.0)
    acc_Z=np.asarray(acc_Zind)

    df2=pd.DataFrame()

    df2['ACC_X (in g)']=pd.Series(acc_X)
    df2['ACC_Y (in g)']=pd.Series(acc_Y)
    df2['ACC_Z (in g)']=pd.Series(acc_Z)
    df2['timestep']=pd.Series(time_step)


    return df2



def find_slop_distance (df , col_name):

    slop_distance = pd.DataFrame(columns=['distance' , 'slop' ,'theta' , 'timestep'])

    for index , row in df.iterrows():

        first_point = row[col_name]
        first_point_moment = row['timestep']
        if index == len(df)-1:
            break

        else :
            second_point_moment = df.loc[index + 1 , 'timestep']
            second_point = df.loc[index + 1 , col_name]  
        distance = abs(second_point - first_point)
        slop  = 0 #(second_point - first_point) / (second_point_moment - first_point_moment)
        theta =  (math.atan2((second_point - first_point) , (second_point_moment - first_point_moment)) * 180 / math.pi)
        slop_distance.loc[len(slop_distance)] = [distance , slop , theta, second_point_moment]

    return slop_distance


def fall_detect (window_to_decimal , threshold_x , threshold_y):
        # print(threshold_y)
        # print(threshold_x)

        fall_event_time  = [0]
        df_x = find_slop_distance(window_to_decimal ,'ACC_X (in g)')
        df_y = find_slop_distance(window_to_decimal , 'ACC_Y (in g)')
        df_z = find_slop_distance(window_to_decimal , 'ACC_Z (in g)')

        df_x_angle = df_x[df_x['theta'] < -75]
        df_y_angle = df_y[df_y['theta'] < -75]
        df_z_angle = df_z[df_z['theta'] < -75]

        df_x_angle = df_x_angle.reset_index()
        df_y_angle = df_y_angle.reset_index()
        df_z_angle = df_z_angle.reset_index()



        if len(df_x_angle) == 0  or len(df_y_angle) == 0 : 
            return fall_event_time
            
        else :
            max_height_x_index = -1 
            min_angle_x_index = -2 
            max_height_y_index = -1 
            min_angle_y_index = -2
            max_height_z_index = -1
            min_angle_z_index = -2

            if not(df_x_angle.empty):

                max_height_x = np.max(df_x_angle['distance'])
                min_angle_x = np.min(df_x_angle['theta'])
                max_height_x_index = df_x_angle['distance'].argmax()
                min_angle_x_index = df_x_angle['theta'].argmin()
                max_height_x_count = df_x_angle[df_x_angle['distance'] == max_height_x]
                min_angle_x_count = df_x_angle[df_x_angle['theta'] == min_angle_x]
                
                if len(max_height_x_count) >=2 or len(min_angle_x_count) >=2 or min_angle_x < -90:
                    return fall_event_time  


            if not(df_y_angle.empty):
                max_height_y = np.max(df_y_angle['distance'])
                min_angle_y = np.min(df_y_angle['theta'])
                max_height_y_index = df_y_angle['distance'].argmax()
                min_angle_y_index = df_y_angle['theta'].argmin()
                max_height_y_count = df_y_angle[df_y_angle['distance'] == max_height_y]
                min_angle_y_count = df_y_angle[df_y_angle['theta'] == min_angle_y]
                
                if len(max_height_y_count) >=2 or len(min_angle_y_count) >= 2 or min_angle_y < -90:
                    return fall_event_time
            
            if not(df_z_angle.empty):

                max_height_z = np.max(df_z_angle['distance'])
                min_angle_z = np.min(df_z_angle['theta'])
                max_height_z_index = df_z_angle['distance'].argmax()
                min_angle_z_index = df_z_angle['theta'].argmin()
                max_height_z_count = df_z_angle[df_z_angle['distance'] == max_height_z]
                min_angle_z_count = df_z_angle[df_z_angle['theta'] == min_angle_z] 

                if len(max_height_z_count) >=2 or len(min_angle_z_count) >=2 or min_angle_z < -90:
                    return fall_event_time       

            if (max_height_x_index == min_angle_x_index) and (max_height_y_index == min_angle_y_index) \
                and (max_height_z_index == min_angle_z_index) and (max_height_x >= threshold_x or max_height_y >= threshold_y):
                time_x = df_x_angle.loc[max_height_x_index , 'timestep']
                time_y = df_y_angle.loc[max_height_y_index , 'timestep']

                if abs(time_x -  time_y) < 20:

                    fall_event_time[0] = time_x
                else:

                    fall_event_time[0] = time_x
                    fall_event_time[0] = time_y

        return fall_event_time



def main(window_df , model):
    threshold_x = 0
    threshold_y = 0
    predict_y = []
    activity = ''
    reshaped_segments = []


    window_df_decimal = convert_to_decimal(window_df)
    window_feature = extract_features(window_df_decimal) 

    reshaped_segments = np.asarray(window_feature, dtype= np.float32).reshape(-1, 10, 47)

    predict_y = model.predict(reshaped_segments)

    window_feature = window_feature.drop(window_feature.index[range(0 ,len(window_feature))])   
    window_feature = window_feature.dropna()  
    fall_event_time = []

    # print(predict_y)
    result_index = predict_y.argmax()
    # print(result_index)
    if result_index == 0 :
        activity = 'Walk'
        threshold_x = 0.5
        threshold_y = 0.5

    elif result_index == 1:
        activity = 'Sit'
        threshold_x = 0.5
        threshold_y = 0.5

    elif result_index == 2:
        activity = 'run'
        threshold_x = 0.5
        threshold_y = 0.6 


    fall_event_time = fall_detect(window_df_decimal , threshold_x , threshold_y)
    if fall_event_time[0] == 0:
        fall_event_time[0] = 'No Fall'
    else:
        for i in range (len(fall_event_time)):
            fall_event_time[i] = str(datetime.timedelta(seconds=fall_event_time[i]))

    return activity , fall_event_time

if __name__ == '__main__':
    main(window_df , model)


