from django.urls import include, path
from rest_framework import routers
from .views import heart_rate_ViewSet, heart_rate_new_ViewSet, proccess_heart_rate_data, Accelerometer_ViewSet, \
    proccess_Accelerometer_data, HeartRateDetail, Accelerometer_new_ViewSet, AccelerometerDetail, AccelerometerNotify, \
    HeartRateNotify, Accelerometer_new_V1_ViewSet, AccelerometerDetail_new,acc_for_android_ViewSet,\
    ppg_for_android_ViewSet,ShareView

router = routers.DefaultRouter()

router.register(r'ppg_for_android', ppg_for_android_ViewSet)
router.register(r'acc_for_android', acc_for_android_ViewSet)
router.register(r'heart_rate_voltage', heart_rate_ViewSet)
router.register(r'heart_rate_voltage_new', heart_rate_new_ViewSet)
router.register(r'heart_rate_voltage_get', proccess_heart_rate_data)
router.register(r'Accelerometer_create', Accelerometer_ViewSet)
router.register(r'Accelerometer_add', Accelerometer_new_ViewSet)
router.register(r'Accelerometer_data_get', proccess_Accelerometer_data)

# router.register(r'ppg_try',PpgPostAPIView)
# router.register(r'Accelerometer_add_V1', Accelerometer_new_V1_ViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('', proccess_heart_rate_data.as_view({'get': 'list'})),
    path('', proccess_Accelerometer_data.as_view({'get': 'list'})),
    path('HeartRateDetail/<str:user_id>/', HeartRateDetail.as_view()),
    path('AccelerometerDetail/<str:user_id>/', AccelerometerDetail.as_view()),
    path('Accelerometer_notify/<str:user_id>/', AccelerometerNotify.as_view()),
    path('HeartRate_notify/<str:user_id>/', HeartRateNotify.as_view()),
    path('Accelerometer_add_V1/<str:user_id>/', Accelerometer_new_V1_ViewSet.as_view()),
    path('AccelerometerDetail_new/', AccelerometerDetail_new.as_view()),
    path('NewView/',ShareView.as_view())
    # path('ppg_try', PpgPostAPIView.as_view(), name='ppg_try')
    # path('trial/',Test.as_view()),
]
