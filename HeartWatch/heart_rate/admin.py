from django.contrib import admin
from .models import *
# Register your models here.


admin.site.register(heart_rate_data)
admin.site.register(PPG_data)
admin.site.register(Accelerometer_data)