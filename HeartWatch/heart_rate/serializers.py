from rest_framework import serializers

from .models import PPG_data, Accelerometer_data, PPG_data_new, Accelerometer_data_new


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