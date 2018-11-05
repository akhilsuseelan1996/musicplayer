from sklearn.externals import joblib
model=joblib.load('model.pkl')
import cv2
import time
import os
import numpy
import vlc
from skimage import io
import matplotlib.pyplot as plt
files=os.listdir('yalefaces')

hp=vlc.MediaPlayer("H:\Anshu\Project1\happy\02 - Zaalima - DownloadMing.SE")
sp=vlc.MediaPlayer("H:\Anshu\Project1\surprised\12 - Tum Ho")

cap=cv2.VideoCapture(0)
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
        
        if out=='surprised':
            hp.stop()
            spf=files=os.listdir('H:\Anshu\Project1\surprised')
            k=numpy.random.randint(0,len(spf))
            sp=vlc.MediaPlayer('H:\Anshu\Project1\happy'+spf[k])
            sp.play()
        elif out=='happy':
            sp.stop()
            hpf=files=os.listdir('H:\Anshu\Project1\happy')
            j=numpy.random.randint(0,len(spf))
            hp=vlc.MediaPlayer('H:\Anshu\Project1\happy'+hpf[j])
            hp.play()
            
    time.sleep(4)
    break
    

    

