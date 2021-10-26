from django.urls import include, path
from rest_framework import routers
from .views import heart_rate_ViewSet, proccess_heart_rate_data, Accelerometer_ViewSet, proccess_Accelerometer_data


router = routers.DefaultRouter()
router.register(r'heart_rate_voltage', heart_rate_ViewSet)
router.register(r'heart_rate_voltage_get', proccess_heart_rate_data)
router.register(r'Accelerometer_create', Accelerometer_ViewSet)
router.register(r'Accelerometer_data_get', proccess_Accelerometer_data)

urlpatterns = [
   path('', include(router.urls)),
   path('', proccess_heart_rate_data.as_view({'get': 'list'})),
   path('', proccess_Accelerometer_data.as_view({'get': 'list'})),
]