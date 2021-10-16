from django.urls import include, path
from rest_framework import routers
from .views import heart_rate_ViewSet, proccess_heart_rate_data


router = routers.DefaultRouter()
router.register(r'heart_rate_voltage', heart_rate_ViewSet)
router.register(r'heart_rate_voltage_get', proccess_heart_rate_data)

urlpatterns = [
   path('', include(router.urls)),
   path('', proccess_heart_rate_data.as_view({'get': 'list'})),
]