import cv2
import os
import numpy
from skimage import io
import matplotlib.pyplot as plt
files=os.listdir('C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\yalefaces\\')
x=[]
xdata=[]
ydata=[]
for i in files:
    if i[10:-4]=='happy' or i[10:-4]=='surprised' or i[10:-4]=='sad'or i[10:-4]=='neutral':
        x.append(i)
        
        img=io.imread("C:\\Users\\ajiin\\Desktop\\Anshu\\Project1\\yalefaces\\"+i)
        fd=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        faces=fd.detectMultiScale(img,1.3,5)
        
        for (cx,cy,w,h) in faces:
            img=cv2.rectangle(img,(cx,cy),(cx+w,cy+h),(0,0,255),3)
            img2=img[cy:cy+h+5,cx:cx+w+2]
            img3=img2.astype(numpy.float32)
            img3=img3/(img3.max())
            img4=cv2.resize(img3,(115,115))
            img5=numpy.ndarray.flatten(img4)
            xdata.append(img5)
            ydata.append(i[10:-4])
               
from sklearn import neighbors #implementing k nearest neighbour algorithm
alg=neighbors.KNeighborsClassifier(n_neighbors=3)
xd=numpy.array(xdata)
yd=numpy.array(ydata)
alg.fit(xd,yd)


testi=xd[25]   
alg.predict(testi.reshape(1,-1)) 
alg.score(xd,yd)   

from sklearn.externals import joblib
joblib.dump(alg,'model.pkl')



plt.imshow(img3,cmap='gray')

cv2.imshow('face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
