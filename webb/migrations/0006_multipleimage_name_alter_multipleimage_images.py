# Generated by Django 4.2 on 2023-04-11 10:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('webb', '0005_alter_multipleimage_images'),
    ]

    operations = [
        migrations.AddField(
            model_name='multipleimage',
            name='name',
            field=models.CharField(default=None, max_length=50),
        ),
        migrations.AlterField(
            model_name='multipleimage',
            name='images',
            field=models.ImageField(default=None, upload_to='images/'),
        ),
    ]