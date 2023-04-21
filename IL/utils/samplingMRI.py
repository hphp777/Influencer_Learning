import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import tarfile

# unzip tar file
# fname = 'C:/Users/hb/Desktop/data/BraTS2021/BraTS2021_00495.tar'   
# ap = tarfile.open(fname)   
# ap.extractall('C:/Users/hb/Desktop/data/BraTS2021/Qualification')      
# ap.close()   
modals = ["flair", "t1", "t1ce", "t2"]
imgs = [nib.load(f"C:/Users/hb/Desktop/data/BraTS2021/Training/BraTS2021_0/BraTS2021_00421_{m}.nii.gz").get_fdata().astype(np.float32)[:, :, 75] for m in ["flair", "t1", "t1ce", "t2"]]
lbl = nib.load("C:/Users/hb/Desktop/data/BraTS2021/Training/BraTS2021_0/BraTS2021_00421_seg.nii.gz").get_fdata().astype(np.uint8)[:, :, 75]
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 15))
for i, img in enumerate(imgs):
    ax[i].imshow(img, cmap='gray')
    ax[i].set_title(modals[i])
    ax[i].axis('off')
ax[-1].imshow(lbl, vmin=0, vmax=4)
ax[-1].axis('off')
plt.tight_layout()            
plt.show()

 


