from rest_framework import serializers

from .models import PPG_data


class heart_rate_Serializer(serializers.ModelSerializer):
   class Meta:
       model = PPG_data
       fields = '__all__'


class heart_rate_get_Serializer(serializers.ModelSerializer):
   class Meta:
       model = PPG_data
       fields = '__all__'
