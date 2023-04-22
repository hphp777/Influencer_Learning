import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys


np.set_printoptions(threshold=sys.maxsize)
value = [0]*10
img = np.load('D:/Data/BraTS2021/2D/Training/Participant1/imgs/95.npy')
label = np.load('D:/Data/BraTS2021/2D/Training/Participant1/labels/95.npy')
for y in range(len(label)):
    for x in range(len(label[0])):
        value[int(label[y][x])] += 1
print(value)

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(label, cmap='gray')
# plt.show()

# modals = ["flair", "t1", "t1ce", "t2"]
# imgs = [nib.load(f"C:/Users/hb/Desktop/data/BraTS2021/3D/Qualification/BraTS2021_0/BraTS2021_00019_{m}.nii.gz").get_fdata().astype(np.float32)[:, :, 75] for m in ["flair", "t1", "t1ce", "t2"]]
# lbl = nib.load("C:/Users/hb/Desktop/data/BraTS2021/3D/Qualification/BraTS2021_0/BraTS2021_00019_seg.nii.gz").get_fdata().astype(np.uint8)[:, :, 75]
# fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
# for i, img in enumerate(imgs):
#     ax[i].imshow(img, cmap='gray')
#     ax[i].set_title(modals[i])
#     ax[i].axis('off')
# ax[-1].imshow(lbl, vmin=0, vmax=4)
# ax[-1].axis('off')
# plt.tight_layout()            
# plt.show()