import numpy
from os import listdir
from os.path import isfile,join
import os

files_name=[f for f in listdir('yalefaces')]

for i in files_name:
    j=i[:9]+'_'+i[10:]+'.png'
    os.rename('F:\MLIoT\ML\Project1\yalefaces\\'+i,'F:\MLIoT\ML\Project1\yalefaces\\'+j)
