from sklearn.externals import joblib
model=joblib.load('model.pkl')
import cv2
import os
import numpy
from skimage import io
import matplotlib.pyplot as plt
files=os.listdir('yalefaces')
url='http://172.20.10.14:8080/shot.jpg'
im=io.imread
im=cv2.resize(im,(320,250))
im=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
plt.imshow(im,cmap='gray')
fd=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces=fd.detectMultiScale(im,1.3,5)
for (cx,cy,w,h) in faces:
    im2=im[cy:cy+h+5,cx:cx+w+2]
    im3=im2.astype(numpy.float32)
    im3=im3/(im3.max())
    im4=cv2.resize(im3,(115,115))
    im5=numpy.ndarray.flatten(im4)
    plt.figure()
    plt.imshow(im4,cmap='gray')
    out=model.predict(im5.reshape(1,-1)) 
    print(out)
