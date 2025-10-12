from django.db import models

# Create your models here.
class Expression(models.Model):
    name = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.name} ({self.percentage}%)"

class Eyes(models.Model):
    side = models.CharField(max_length=50)  
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.side} Eye ({self.percentage}%)"

class HandsExpression(models.Model):
    move = models.CharField(max_length=100, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.move} ({self.percentage}%)"

class Speech(models.Model):
    word = models.CharField(max_length=255, unique=True)
    percentage = models.FloatField()

    def _str_(self):
        return f"{self.word} ({self.percentage}%)"