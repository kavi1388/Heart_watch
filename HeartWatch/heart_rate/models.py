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


class PPG_data_new(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500, blank=True)
    heart_rate_voltage = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)

####################  android api ################################

class PPG_data_from_Android(models.Model):
    id = models.AutoField(primary_key=True)
    heart_rate_voltage = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)


class Accelerometer_data_from_Android(models.Model):
    id = models.AutoField(primary_key=True)
    Accelerometer = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)
#################################################################

class Accelerometer_data(models.Model):
    id = models.AutoField(primary_key=True)
    Accelerometer = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)


class Accelerometer_data_new(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500, blank=True)
    Accelerometer = jsonfield.JSONField()

    def __str__(self):
        return str(self.id)


class Accelerometer_result_save(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500, blank=True)
    final_result = models.TextField(blank=True)

    def __str__(self):
        return str(self.id)


class PPG_result_save(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.CharField(max_length=500, blank=True)
    final_result = models.TextField(blank=True)


    def __str__(self):
        return str(self.id)