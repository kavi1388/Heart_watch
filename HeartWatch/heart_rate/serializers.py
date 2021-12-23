from rest_framework import serializers

from .models import PPG_data, Accelerometer_data, PPG_data_new, Accelerometer_data_new, Accelerometer_result_save, \
    PPG_result_save,PPG_data_from_Android,Accelerometer_data_from_Android


####################### Android API ###########################
class ppg_data_android_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_data_from_Android
        fields = '__all__'


class Accelerometer_data_android_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_data_from_Android
        fields = '__all__'

class Ppg_New_Serializer(serializers.Serializer):
    text = serializers.CharField()
    def calculate(self,attrs):
        result = 'hello'
        return result

###############################################################
class heart_rate_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_data
        fields = '__all__'


class heart_rate_get_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_data
        fields = '__all__'


#########################################################################################
class heart_rate_new_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_data_new
        fields = '__all__'


class heart_rate_get_new_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_data_new
        fields = '__all__'


#########################################################################################
class Accelerometer_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_data
        fields = '__all__'


class Accelerometer_get_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_data
        fields = '__all__'


#########################################################################################
class Accelerometer_new_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_data_new
        fields = '__all__'


class Accelerometer_get_new_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_data_new
        fields = '__all__'


class Accelerometer_notify_Serializer(serializers.ModelSerializer):
    class Meta:
        model = Accelerometer_result_save
        fields = '__all__'


class HeartRate_notify_Serializer(serializers.ModelSerializer):
    class Meta:
        model = PPG_result_save
        fields = '__all__'
