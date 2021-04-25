import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
path = 'face_dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = os.path.split(imagePath)[-1].split(".")[0]
        faces = face_cascade.detectMultiScale(img_numpy)
        for x, y, w, h in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
label_encoder = LabelEncoder()

unique = np.unique(ids)
for i in range(len(ids)):
    ids[i] = int(np.where(unique == ids[i])[0])

ids = ids.astype(int)

ids = label_encoder.fit_transform(ids)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer.yml') 
