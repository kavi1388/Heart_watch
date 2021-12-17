from django.contrib import admin
from .models import *
from import_export.admin import ExportActionMixin


# Register your models here.

class PPG_data_from_AndroidAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'heart_rate_voltage')


class acc_for_android_ViewSet(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'heart_rate_voltage')


# Register your models here.
class PPG_DataAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'heart_rate_voltage')


class Accelerometer_dataAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'Accelerometer')


class PPG_data_newAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'user_id', 'heart_rate_voltage')


class Accelerometer_data_newAdmin(ExportActionMixin, admin.ModelAdmin):
    list_display = ('id', 'user_id', 'Accelerometer')


admin.site.register(PPG_result_save)
admin.site.register(Accelerometer_result_save)
admin.site.register(PPG_data, PPG_DataAdmin)
admin.site.register(PPG_data_new, PPG_data_newAdmin)
admin.site.register(Accelerometer_data, Accelerometer_dataAdmin)
admin.site.register(Accelerometer_data_new, Accelerometer_data_newAdmin)
