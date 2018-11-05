from tkinter import *
root=Tk()
from sklearn.externals import joblib       #calling the testcode_1 to access the models
model=joblib.load('model.pkl')

import cv2
import time
import os
import numpy
from skimage import io
from pygame import mixer                  #importing pygame for media playing
import matplotlib.pyplot as plt

files=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\yalefaces\\')

mixer.init()                              #initializing mixer module


l1=Label(root,text="Mood Player").grid(row=0, column=0)
l2=Label(root,text="Mood Selector").grid(row=1,column=0)

def happysong():
    sspf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\surprised\\')
    os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\surprised\\')
    k=numpy.random.randint(0,len(sspf))                       #to play random songs
    mixer.music.load(sspf[k])
    mixer.music.play()

def angrysong():
    hhpf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\happy\\')
    os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\happy\\')
    j=numpy.random.randint(0,len(hhpf))                     #to play random songs
    mixer.music.load(hhpf[j])
    mixer.music.play()

def sadsong():
    aapf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\sad\\')
    os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\sad\\')
    m=numpy.random.randint(0,len(aapf))                    #to play random songs
    mixer.music.load(aapf[m])
    mixer.music.play()

def neutralsong():
    nnpf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\neutral\\')
    os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\neutral\\')
    n=numpy.random.randint(0,len(nnpf))                    #to play random songs
    mixer.music.load(nnpf[n])
    mixer.music.play()
    
    
happyButton=Button(root,height=2,width=5, text="happy", command= happysong ).grid(row=2, column=0,padx=5,pady=5)
angryButton=Button(root,height=2,width=5, text="angry", command=angrysong ).grid(row=2, column=1,padx=30,pady=5)
sadButton=Button(root,height=2,width=5, text="sad", command=sadsong ).grid(row=2, column=2,padx=30,pady=5)
neutralButton=Button(root,height=2,width=5, text="neutral",command=neutralsong ).grid(row=2, column=3,padx=30,pady=5)

l3=Label(root,text="Live Emotion Capture").grid(row=3, column=0)



def mlfunction():


    cap=cv2.VideoCapture(0)                   #accessing webcam

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
                spf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\surprised\\')
                os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\surprised\\')
                k=numpy.random.randint(0,len(spf))                       #to play random songs
                mixer.music.load(spf[k])
                mixer.music.play()
            
            
            elif out=='happy':
                hpf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\happy\\')
                os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\happy\\')
                j=numpy.random.randint(0,len(hpf))                     #to play random songs
                mixer.music.load(hpf[j])
                mixer.music.play()

            elif out=='sad':
                apf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\sad\\')
                os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\sad\\')
                m=numpy.random.randint(0,len(apf))                    #to play random songs
                mixer.music.load(apf[m])
                mixer.music.play()

            elif out=='neutral':
                npf=os.listdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\neutral\\')
                os.chdir('C:\\Users\\user\\Desktop\\Anshu\\Project1\\neutral\\')
                n=numpy.random.randint(0,len(npf))                    #to play random songs
                mixer.music.load(npf[n])
                mixer.music.play()

            
            
        time.sleep(4)
        break

cameraButton=Button(root,height=2,width=10, text="Open Camera", command= mlfunction ).grid(row=4, column=0, pady=5)
