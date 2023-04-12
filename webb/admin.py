from django.contrib import admin
from .models import MultipleImage

class MemberAdmin(admin.ModelAdmin):
    list_display = ("id","images")
# Register your models here.
admin.site.register(MultipleImage,MemberAdmin)
