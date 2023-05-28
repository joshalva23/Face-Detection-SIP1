import cv2 as cv
import os
import numpy as np
from django.http import HttpResponse
from web.settings import MEDIA_ROOT

DIR = MEDIA_ROOT + "\\images"
def live_detect(request):

    def resize(frame, scale=0.4):
        dimension = (int(frame.shape[1]*scale), int(frame.shape[0]*scale))
        return cv.resize(frame, dimension)


    people = []

    for i in os.listdir(DIR):
        people.append(i)

    # DIR = r'C:\Users\Rohith gowda M\OneDrive\Desktop\New folder'
    # feats = []
    # labels = []

    haar_cas = cv.CascadeClassifier('haar_face.xml')


    # def unk():
    #     for i in people:
    #         img_path = os.path.join(DIR, i)
    #         label = people.index(i)

    #         for img in os.listdir(img_path):
    #             im_path = os.path.join(img_path, img)

    #             im = cv.imread(im_path)
    #             gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    #             mod = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    #             for (k, l, m, n) in mod:
    #                 det = gray[l:l+n, k:k+m]

    #                 feats.append(det)
    #                 labels.append(label)


    # unk()


    # faces_mode = cv.face.LBPHFaceRecognizer_create()

    # feats = np.array(feats, dtype='object')
    # labels = np.array(labels)

    # faces_mode.train(feats, labels)

    # faces_mode.save('.yml')

    features = np.load('features.npy', allow_pickle=True)
    labels = np.load('labels.npy', allow_pickle=True)

    face_rec = cv.face.LBPHFaceRecognizer_create()
    face_rec.read('trained_face.yml')

    # img = cv.imread(r'C:\Users\M P Akash\Desktop\Pic\train\Akash\20210724_160337.jpg')

    """


    img1 = resize(img, scale=0.3)
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    detr = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (q, w, e, r) in detr:
        dt = gray[w:w+r, q:q+r]

        label, confi = faces_mode.predict(dt)
        print(f'The image is {people[label]} with confidence of {confi}')

        cv.rectangle(img1, (q, w), (q+e, w+r), (0, 255, 0), thickness=2)
        cv.putText(img1, str(people[label]), (50, 50), cv.FONT_HERSHEY_DUPLEX, 1., (0, 255, 0), thickness=2)

    cv.imshow('DEt', img1)
    cv.waitKey(0)
    """


    cap = cv.VideoCapture(0)

    while True:
        istrue, frame = cap.read()

        frame1 = cv.flip(frame, 1)
        gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        hr = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        for (a, s, d, f) in hr:
            dew = gray[s:s+f, a:a+d]

            label, confi = face_rec.predict(dew)
            print(f'The image is {people[label]} with confidence {confi}')

            cv.rectangle(frame1, (a, s), (a+d, s+f), (0, 255, 0), thickness=5)
            cv.putText(frame1, str(people[label]), (40, 40), cv.FONT_HERSHEY_DUPLEX, 1., (225, 0, 0), thickness=2)

        cv.imshow('Mukha', frame1)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv.destroyAllWindows()
    return HttpResponse(1)

def human_face(request):

    # Load the Haar Cascade Classifier
    cascade_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Start the video capture
    video_capture = cv.VideoCapture(0)

    # Loop over the frames of the video
    while True:
        # Read the current frame from the video stream
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('Video', frame)

        # Exit the loop if the 'q' key is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video_capture.release()
    cv.destroyAllWindows()
    return HttpResponse(1)