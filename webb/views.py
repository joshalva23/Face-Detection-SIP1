from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.http import HttpResponse,JsonResponse
from django.contrib.auth import logout
from django.template import loader
from os import remove
from web.settings import MEDIA_ROOT

#import logging
#logger = logging.getLogger('django')
# Create your views here.
def index(request):
    if request.user.is_anonymous:
        return redirect("/login")
    return render(request,'Signed_in.html')

def loginuser(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            # A backend authenticated the credentials
            login(request,user)
            return redirect("/")
        else:
            # No backend authenticated the credentials
            return render(request,'login.html')
    return render(request,'login.html')

def logoutUser(request):
    logout(request)
    return redirect("/login")


def about(request):
    return render(request,'about.html')


from .models import MultipleImage
def upload(request):
    if request.method == "POST":
        images = request.FILES.getlist('images')
        name = request.POST['folder']
        for image in images:
            #if( MultipleImage.objects.filter(images=image).values_list('images',flat=True) is None ):
            MultipleImage.objects.create(images=image,name=name)
    images = MultipleImage.objects.all()
    return render(request, 'Upload.html', {'images': images})
    

def reload():
    return

def deleteimage(request):
    path = request.POST['imagepath']
    print("----",path)
    image = MultipleImage.objects.filter(images=path)
    print("----",'Loading')
    if image is not None:
        image.delete()
    print("----",MEDIA_ROOT+"\\"+path[0:len("images")]+"\\"+path[len("images")+1:])
    remove(MEDIA_ROOT+"\\"+path[0:len("images")]+"\\"+path[len("images")+1:])
    return redirect('/upload')
    
def train(request):
    return HttpResponse(render(request,'Train.html'))

def detect(request):
    return HttpResponse(render(request,'Video_Detect.html'))