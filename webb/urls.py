from django.contrib import admin
from django.urls import path
from webb import views,trainmodel,live
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("",views.index, name='webb'),
    path("about",views.about, name='about'),
    path("login",views.loginuser, name='login'),
    path("logout",views.logoutUser, name='logout'),
    path('upload', views.upload, name='upload'),
    path("reload",views.reload, name='reload' ),
    path("train",views.train,name="train"),
    path("training",trainmodel.train, name='training'),
    path('deleteimage',views.deleteimage,name='deleteimage'),
    path('detectface',live.live_detect,name='detectface'),
    path('detecthumanface',live.human_face,name='detecthumanface'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)