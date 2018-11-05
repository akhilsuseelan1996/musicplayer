from sklearn.externals import joblib #calling the testcode_1 to access the models
model=joblib.load('model.pkl')

import cv2
import time
import os
import numpy
from skimage import io
from pygame import mixer  #importing pygame for media playing
import matplotlib.pyplot as plt

files=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\yalefaces\\')

mixer.init() #initializing mixer module

cap=cv2.VideoCapture(0) #accessing webcam

while True:
    ret,im=cap.read()
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
        
        #finding out the emotion
        if out=='surprised':
            spf=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\surprised\\')
            os.chdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\surprised\\')
            k=numpy.random.randint(0,len(spf)) #to play random songs
            mixer.music.load(spf[k])
            mixer.music.play()
            
            
        elif out=='happy':
            hpf=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\happy\\')
            os.chdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\happy\\')
            j=numpy.random.randint(0,len(hpf)) #to play random songs
            mixer.music.load(hpf[j])
            mixer.music.play()

        elif out=='sad':
            apf=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\sad\\')
            os.chdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\sad\\')
            m=numpy.random.randint(0,len(apf)) #to play random songs
            mixer.music.load(apf[m])
            mixer.music.play()

        elif out=='neutral':
            npf=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\neutral\\')
            os.chdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\neutral\\')
            n=numpy.random.randint(0,len(npf)) #to play random songs
            mixer.music.load(npf[n])
            mixer.music.play()

            
            
    time.sleep(4)
    break
    

    

