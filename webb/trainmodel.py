import os
import cv2 as cv
import numpy as np
from web.settings import MEDIA_ROOT
from django.http import HttpResponse

DIR = MEDIA_ROOT + "\\images"
people = [f.path for f in os.scandir(DIR) if f.is_dir()]

def train(request):
    people = []
    print(DIR)
    '''
    for path in Path(DIR).iterdir():
        if path.is_dir():
            people += path
    '''
    people = [f.path for f in os.scandir(DIR) if f.is_dir()]

    for i in range(len(people)):
        people[i] = people[i][len(DIR)+1:]
    print(people)
    # pep = []
    # for i in os.listdir(r'C:\Users\M P Akash\Desktop\Pic\train'):
    #    pep.append(i)

    #DIR = r"C:\Users\hp\Pictures\FO"
    
    features = []
    labels = []

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    def train_img():
        for person in people:
            path = os.path.join(DIR, person)
            label = people.index(person)

            for img in os.listdir(path):
                img_path = os.path.join(path, img)

                img_array = cv.imread(img_path)
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

                faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

                for (i, j, k, l) in faces_rect:
                    faces_cr = gray[j:j+l, i:i+k]

                    features.append(faces_cr)
                    labels.append(label)


    train_img()

    print(f'length of features = {len(features)}')
    print('Training done -->')

    # training recognizer using features and labels

    # instantiate face recognizer
    face_rec = cv.face.LBPHFaceRecognizer_create()

    features = np.array(features, dtype='object')
    labels = np.array(labels)
    print(labels)

    # train the recognizer
    face_rec.train(features, labels)

    np.save('features.npy', features)
    np.save('labels.npy', labels)

    face_rec.save('trained_face.yml')
    return HttpResponse(1)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2

# Load the trained LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_face.yml')

@csrf_exempt
def recognize_face(request):
    if request.method == 'POST':
        # Get the image data from the request
        image_data = request.FILES.get('image').read()

        # Convert the image data to a NumPy array
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Perform face recognition on the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        id_, confidence = recognizer.predict(gray)

        # Return the result as JSON
        result = {'id': id_, 'confidence': confidence}
        return JsonResponse(result)

    else:
        return JsonResponse({'error': 'Invalid request method'})
