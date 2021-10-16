from django.db import models
import jsonfield


# Create your models here.


class heart_rate_data(models.Model):
    id = models.AutoField(primary_key=True)
    heart_rate_voltage = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)


class PPG_data(models.Model):
    id = models.AutoField(primary_key=True)
    heart_rate_voltage = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)
