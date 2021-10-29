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
                if t_diff_afib[i + 1] - t_diff_afib[i] > 10:
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
            return 0


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
        Accelerometer_instance = Accelerometer_data.objects.all().order_by('-id')[:10]

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

        dd = {
            "time": time_last,
            "activity": activity,
            "fall": fall
        }
        Accelerometer_result_save.objects.create(final_result=dd, user_id=user_id)
        return Response(dd)


class AccelerometerNotify(APIView):

    def get_object(self, user_id):
        # Returns an object instance that should
        # be used for detail views.
        try:
            return Accelerometer_result_save.objects.filter(user_id=user_id).order_by('-id')[:1]
        except Accelerometer_result_save.DoesNotExist:
            raise Http404

    def get(self, request, user_id, format=None):
        Accelerometer_obj = self.get_object(user_id)
        # print("Accelerometer_obj ::", Accelerometer_obj)
        serializer = Accelerometer_notify_Serializer(Accelerometer_obj, many=True)
        Accelerometer_insta = serializer.data
        print(Accelerometer_insta)

        return Response(Accelerometer_insta)


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
            return PPG_data_new.objects.filter(user_id=user_id).order_by('-id')[:30]
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
        result = self.ailments_stats(heart_rate_data_list)
        PPG_result_save.objects.create(final_result=result, user_id=user_id)
        return Response(result)

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
            final_pr, ppg_21, ppg_sig, ppg_bpf, t_diff_afib, hr_extracted, peaks_all2, non_uniform = ppg_plot_hr(
                ppg_sig, time_val, fl=0.2, fh=3.5, o=4, n=6, diff_max=4, r=1)

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
        print(HeartRate_insta)

        return Response(HeartRate_insta)
