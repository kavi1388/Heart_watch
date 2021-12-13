import requests
from rest_framework import viewsets, status
from .serializers import heart_rate_Serializer, heart_rate_get_Serializer, Accelerometer_Serializer, \
    Accelerometer_get_Serializer, heart_rate_new_Serializer, heart_rate_get_new_Serializer, \
    Accelerometer_new_Serializer, Accelerometer_get_new_Serializer, Accelerometer_notify_Serializer, \
    HeartRate_notify_Serializer
from .models import PPG_data, Accelerometer_data, PPG_data_new, Accelerometer_data_new, PPG_result_save, \
    Accelerometer_result_save
from rest_framework.response import Response
from .PPG.ppg_hr import *
import datetime
import time
import json
from .PPG.custom_modules import decimal_to_binary
from .PPG.rr_from_ppg import rr_calulation
from .Accelerometer.HAR_Fall import call_model
from rest_framework.views import APIView
from django.http import Http404


# Create your views here.


class heart_rate_ViewSet(viewsets.ModelViewSet):
    queryset = PPG_data.objects.all()
    serializer_class = heart_rate_Serializer

    def post(self, request, format=None):
        serializer = heart_rate_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                            status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class proccess_heart_rate_data(viewsets.ModelViewSet):
    queryset = PPG_data.objects.all()

    def get_serializer_class(self):
        return heart_rate_get_Serializer

    def list(self, request, *args, **kwargs):
        heart_rate_data_list = []
        ppg_instance = PPG_data.objects.all().order_by('-id')[:60]

        serializer = heart_rate_get_Serializer(ppg_instance, many=True)
        heart_rate_insta = serializer.data
        # print(heart_rate_insta)

        for i in heart_rate_insta:
            gg = i['heart_rate_voltage']
            heart_rate_data_list.append(gg)

        # call ailments_stats method
        result = self.ailments_stats(heart_rate_data_list)

        heart_rate = {
            "afib_in": result[0],
            "tachy_in": result[1],
            "brady_in": result[2]

        }
        return Response(heart_rate)

    def ailments_stats(self, ppg_list):

        strike = 0
        strike_tachy = 0
        strike_afib = 0
        count = 20
        count_afib = 10
        brady_in = False
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
            final_pr, ppg_21, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted = ppg_plot_hr(ppg_sig, time_val)

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

            for i in range(len(t_diff_afib) - 1):
                if t_diff_afib[i + 1] - t_diff_afib[i] > 30:
                    strike_afib += 1
                else:
                    strike_afib = 0
                if strike_afib == count_afib:
                    # print('Patient has Atrial Fibrillation')
                    afib_in = True

            # One API call for Atrial Fibrillation
            # return ppg_sig, hr_extracted, final_pr, afib_in, tachy_in, brady_in, data_valid
            return afib_in, tachy_in, brady_in

        else:
            statement = 'Data missing for over 2 minutes , PPG analysis not done'
            return statement


# Accelerometer Data insert
class Accelerometer_ViewSet(viewsets.ModelViewSet):
    queryset = Accelerometer_data.objects.all()
    serializer_class = Accelerometer_Serializer

    def post(self, request, format=None):
        serializer = Accelerometer_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                            status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class proccess_Accelerometer_data(viewsets.ModelViewSet):
    queryset = Accelerometer_data.objects.all()

    def get_serializer_class(self):
        return Accelerometer_Serializer

    def list(self, request, *args, **kwargs):
        Accelerometer_list = []
        Accelerometer_instance = Accelerometer_data.objects.all().order_by('-id')[:30]

        serializer = Accelerometer_Serializer(Accelerometer_instance, many=True)
        Accelerometer_insta = serializer.data
        # print(heart_rate_insta)

        for i in Accelerometer_insta:
            gg = i['Accelerometer']
            Accelerometer_list.append(gg)

        # print("Accelerometer_list ::", Accelerometer_list)

        # call ailments_stats method
        activity, fall = call_model(Accelerometer_list)

        dd = {
            "activity": activity,
            "fall": fall,
            "check": 'new'
        }
        return Response(dd)


############################################## NEW Accelerometer API Start #############################################
class Accelerometer_new_ViewSet(viewsets.ModelViewSet):
    queryset = Accelerometer_data_new.objects.all()
    serializer_class = Accelerometer_new_Serializer

    def post(self, request, format=None):
        serializer = Accelerometer_new_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                            status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AccelerometerDetail(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return Accelerometer_data_new.objects.filter(user_id=user_id).order_by('-id')[:10]
        except Accelerometer_data_new.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        Accelerometer_data_list = []
        Accelerometer_obj = self.get_object(user_id)
        serializer = Accelerometer_get_new_Serializer(Accelerometer_obj, many=True)
        Accelerometer_insta = serializer.data
        for i in Accelerometer_insta:
            gg = i['Accelerometer']
            Accelerometer_data_list.append(gg)
        time_last, activity, fall = call_model(Accelerometer_data_list)
        if fall[0][0] == 'No Fall':
            api_type = None
        else:
            #One API for Fall with type 3==True
            api_type = 3

        dd = {
            "time": time_last,
            "activity": activity,
            "fall": fall
        }
        Accelerometer_result_save.objects.create(final_result=dd, user_id=user_id)
        return Response(dd)


class AccelerometerDetail_new(APIView):
    serializer_class = Accelerometer_new_Serializer, Accelerometer_get_new_Serializer
    def post(self, request, format=None):
        Accelerometer_data_list = []
        Accelerometer_data = ""
        serializer1 = Accelerometer_new_Serializer(data=request.data)
        if serializer1.is_valid():
            serializer1.save()
            acc_obj = serializer1.data
            user_id = acc_obj['user_id']
            # user_id = "6052e4dc605f500004ef6d3f"
            first_ten = Accelerometer_data_new.objects.filter(user_id=user_id).order_by('-id')[:30]
            # print(first_ten)
            serializer = Accelerometer_get_new_Serializer(first_ten, many=True)
            Accelerometer_insta = serializer.data
            # print(Accelerometer_insta)
            for i in Accelerometer_insta:
                gg = i['Accelerometer']
                Accelerometer_data_list.append(gg)
            time_last, activity, fall = call_model(Accelerometer_data_list[-1::-1])
            if fall[0][0] == 'No Fall':
                api_type=None
            else:
                # One API for Fall with type 3==True
                api_type=3

            dd = {
                "time": time_last,
                "activity": activity,
                "fall": fall
            }

            last_time = dd['time']
            # print(last_time)
            current_time = time.strftime('%H:%M:%S', time.localtime())
            x = time.strptime(current_time, '%H:%M:%S')
            y = time.strptime(last_time, '%H:%M:%S')
            time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                           seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                  minutes=y.tm_min,
                                                                                                  seconds=y.tm_sec).total_seconds()
            if abs(time_diff) > 120:
                Accelerometer_data = {
                    "final_result": 'No activity detected'
                }
            else:
                activity = dd['activity'][0]
                Accelerometer_data = {
                    'activity': activity
                }
            return Response(Accelerometer_data,
                            status=status.HTTP_200_OK)
        return Http404

class Accelerometer_new_V1_ViewSet(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return Accelerometer_data_new.objects.filter(user_id=user_id).order_by('-id')[:30]
        except Accelerometer_data_new.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        User_alert_url = 'http://164.52.214.242:9098/user-alerts?alertsToken=M0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ'
        User_activity_url = 'http://164.52.214.242:9098/user-activity'
        Accelerometer_data_list = []
        Accelerometer_obj = self.get_object(user_id)
        serializer = Accelerometer_get_new_Serializer(Accelerometer_obj, many=True)
        Accelerometer_insta = serializer.data
        for i in Accelerometer_insta:
            gg = i['Accelerometer']
            Accelerometer_data_list.append(gg)
        time_last, activity, fall = call_model(Accelerometer_data_list[-1::-1])

        User_activity_url = 'http://164.52.214.242:9098/user-activity'
        activityObj = {"userID": user_id, "activityType": activity[0], "timestamp": time.strftime('%d/%m/%Y'),
                       "duration": "10"}
        act_res = requests.post(User_activity_url, json=activityObj)

        if fall[0][0] == 'No Fall':
            api_type= None
        else:
            #One API for Fall with type 3==True
            record_time = result['Time Interval'][-1]
            current_time = time.strftime('%H:%M:%S', time.localtime())
            x = time.strptime(current_time, '%H:%M:%S')
            y = time.strptime(record_time, '%H:%M:%S')
            time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                           seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                  minutes=y.tm_min,
                                                                                                  seconds=y.tm_sec).total_seconds()
            if time_diff < 30:
                api_type = "3"
                accelerometer_obj = {"userID": user_id, "alertType": api_type}
                res = requests.post(User_alert_url, json=accelerometer_obj)

            # print(res.text)
        dd = {
            "time": time_last,
            "activity": activity,
            "fall": fall
        }
        Accelerometer_result_save.objects.create(final_result=dd, user_id=user_id)
        Accelerometer_data = ""
        accel_notify = Accelerometer_result_save.objects.filter(user_id=user_id).order_by('-id')[:1]
        serializer = Accelerometer_notify_Serializer(accel_notify, many=True)
        Accelerometer_insta = serializer.data

        try:
            for i in Accelerometer_insta:
                final_result = i['final_result']
                ff = eval(final_result)
                d = dict(ff)
                last_time = d['time']

                current_time = time.strftime('%H:%M:%S', time.localtime())
                x = time.strptime(current_time, '%H:%M:%S')
                y = time.strptime(last_time, '%H:%M:%S')
                time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                               seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                      minutes=y.tm_min,
                                                                                                      seconds=y.tm_sec).total_seconds()
                if abs(time_diff) > 120:
                    Accelerometer_data = {
                        "time":last_time,
                        "current time": current_time,
                        "time diff": time_diff,
                        "final_result": 'No activity detected'
                    }
                else:
                    activity = d['activity'][0]
                    Accelerometer_data = {
                        'time': last_time,
                        'activity': activity
                    }
            return Response(Accelerometer_data)
        except:
            return Http404
        # return Response(dd)

class AccelerometerNotify(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return Accelerometer_result_save.objects.filter(user_id=user_id).order_by('-id')[:1]
        except Accelerometer_result_save.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        Accelerometer_data = ""
        Accelerometer_obj = self.get_object(user_id)
        serializer = Accelerometer_notify_Serializer(Accelerometer_obj, many=True)
        Accelerometer_insta = serializer.data
        try:
            for i in Accelerometer_insta:
                final_result = i['final_result']
                ff = eval(final_result)
                d = dict(ff)
                last_time = d['time']

                current_time = time.strftime('%H:%M:%S', time.localtime())
                x = time.strptime(current_time, '%H:%M:%S')
                y = time.strptime(last_time, '%H:%M:%S')
                time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                               seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                      minutes=y.tm_min,
                                                                                                      seconds=y.tm_sec).total_seconds()
                if abs(time_diff) > 120:
                    Accelerometer_data = {
                      "final_result": 'No activity detected'
                    }
                else:
                    activity = d['activity'][0]
                    Accelerometer_data = {
                        'activity': activity
                    }
            return Response(Accelerometer_data)
        except :
            return Http404

############################################## NEW Heart rate API Start ################################################


class heart_rate_new_ViewSet(viewsets.ModelViewSet):
    queryset = PPG_data_new.objects.all()
    serializer_class = heart_rate_new_Serializer

    def post(self, request, format=None):
        serializer = heart_rate_new_Serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                            status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class HeartRateDetail(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return PPG_data_new.objects.filter(user_id=user_id).order_by('-id')[20:65]
        except PPG_data_new.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        heart_rate_data_list = []
        heart_rate_obj = self.get_object(user_id)
        serializer = heart_rate_get_new_Serializer(heart_rate_obj, many=True)
        heart_rate_insta = serializer.data
        for i in heart_rate_insta:
            gg = i['heart_rate_voltage']
            heart_rate_data_list.append(gg)
            # call ailments_stats method

        result = self.ailments_stats(heart_rate_data_list[-1::-1])
        api_type=None
        if type(result) is not str:
            User_alert_url = 'http://164.52.214.242:9098/user-alerts?alertsToken=M0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ'
            if result['tachycardia']:
                record_time=result['Time Interval'][-1]
                current_time = time.strftime('%H:%M:%S', time.localtime())
                x = time.strptime(current_time, '%H:%M:%S')
                y = time.strptime(record_time, '%H:%M:%S')
                time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                               seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                      minutes=y.tm_min,
                                                                                                      seconds=y.tm_sec).total_seconds()
                if time_diff < 30:
                    api_type = "2"
                    hr_obj = {"userID": user_id, "alertType": api_type}
                    res = requests.post(User_alert_url, json=hr_obj)
            elif result['bradycardia']:
                record_time = result['Time Interval'][-1]
                current_time = time.strftime('%H:%M:%S', time.localtime())
                x = time.strptime(current_time, '%H:%M:%S')
                y = time.strptime(record_time, '%H:%M:%S')
                time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                               seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                      minutes=y.tm_min,
                                                                                                      seconds=y.tm_sec).total_seconds()
                if time_diff < 30:
                    api_type = "1"
                    hr_obj = {"userID": user_id, "alertType": api_type}
                    res = requests.post(User_alert_url, json=hr_obj)

        if result['a_Fib']:
            record_time = result['Time Interval'][-1]
            current_time = time.strftime('%H:%M:%S', time.localtime())
            x = time.strptime(current_time, '%H:%M:%S')
            y = time.strptime(record_time, '%H:%M:%S')
            time_diff = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                           seconds=x.tm_sec).total_seconds() - datetime.timedelta(hours=y.tm_hour,
                                                                                                  minutes=y.tm_min,
                                                                                                  seconds=y.tm_sec).total_seconds()
            if time_diff < 30:
                api_type = "5"
                hr_obj = {"userID": user_id, "alertType": api_type}
                res = requests.post(User_alert_url, json=hr_obj)


            # print(res.text)
        PPG_result_save.objects.create(final_result=result, user_id=user_id)
        return Response(result)

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
                ppg_sig, time_val,  fl=1, fh=5, o=4, n=5, diff_max=10, r=5)
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



            res = {'time_interval': (time_val[0], time_val[-1]), 'predicted_HR': hr_extracted, 'predicted_SPO2':spo2_pred,
                   'resp_rate':resp_rate,'rr_peak_intervals': t_diff_afib, 'a_Fib': afib_in, 'tachycardia': tachy_in,
                   'bradycardia': brady_in, 'dataFrame': final_pr, 'api_type': api_type}

            return res
        else:
            statement = 'Data missing for over 2 minutes , PPG analysis not done'
            return statement


class HeartRateNotify(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return PPG_result_save.objects.filter(user_id=user_id).order_by('-id')[:1]
        except PPG_result_save.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        HeartRate_obj = self.get_object(user_id)
        # print("HeartRate_obj ::", HeartRate_obj)
        serializer = HeartRate_notify_Serializer(HeartRate_obj, many=True)
        HeartRate_insta = serializer.data
        # print(HeartRate_insta)

        return Response(HeartRate_insta)
