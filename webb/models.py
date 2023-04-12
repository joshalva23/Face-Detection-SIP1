from django.db import models

def attachment_path(instance,filename):
    return f'images/{instance.name}/{instance.images}'
# Create your models here.
class MultipleImage(models.Model):
    name = models.CharField(max_length=50, default=None)
    images = models.ImageField(upload_to=attachment_path, default=None)