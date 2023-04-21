import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import tarfile
import random
import os
import shutil

folders = os.listdir('C:/Users/hb/Desktop/data/BraTS2021/Data')

datas = list(range(1251))
random.shuffle(datas)

t1 = int(0.8*len(datas))
t2 = int(0.9*len(datas))
training = datas[:t1]
qualification = datas[t1:t2]
test = datas[t2:]

print(test)

for i in range(len(training)):
    src = "C:/Users/hb/Desktop/data/BraTS2021/Data/BraTS2021_" + str(training[i])
    dst = "C:/Users/hb/Desktop/data/BraTS2021/Training/BraTS2021_" + str(i)
    shutil.copytree(src, dst)

for i in range(len(qualification)):
    src = "C:/Users/hb/Desktop/data/BraTS2021/Data/BraTS2021_" + str(qualification[i])
    dst = "C:/Users/hb/Desktop/data/BraTS2021/Qualification/BraTS2021_" + str(i)
    shutil.copytree(src, dst)

for i in range(len(test)):
    src = "C:/Users/hb/Desktop/data/BraTS2021/Data/BraTS2021_" + str(test[i])
    dst = "C:/Users/hb/Desktop/data/BraTS2021/Test/BraTS2021_" + str(i)
    shutil.copytree(src, dst)