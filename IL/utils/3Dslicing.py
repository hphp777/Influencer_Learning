import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os

modals = ["flair", "t1", "t1ce", "t2"]
p1 = list(range(200))
p2 = list(range(200,400))
p3 = list(range(400,600))
p4 = list(range(600,800))
p5 = list(range(800,1000))
q = list(range(125))
t = list(range(126))

folder_path = "C:/Users/hb/Desktop/data/BraTS2021/3D/Training/BraTS2021_" ### 
save_path_img = "C:/Users/hb/Desktop/data/BraTS2021/2D/Training/Participant4/imgs/" ###
save_path_label = "C:/Users/hb/Desktop/data/BraTS2021/2D/Training/Participant4/labels/" ###
data_idx = 0

for i in range(len(p4)): #### 
    item_path = folder_path + str(p4[i]) ### 
    items = os.listdir(item_path)
    label = None
    label_index = None
    for j in range(len(items)):
        if 'seg' in items[j]:
            label = items[j]
            label_index = j
    items.pop(label_index)
    label_path = item_path + '/' + label
    for j in range(len(items)):
        img = nib.load(item_path + '/' + items[j]).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)
        # Save 3D to 2D mat
        for z in range(155): # z axis
            np.save(save_path_img + str(data_idx), img[:,:,z])
            np.save(save_path_label + str(data_idx), label[:,:,z])
            data_idx += 1

# 1 ~ 200, 201 ~ 400, 401 ~ 600, 601 ~ 800, 801 ~ 1000


