import torch 
from models.unet_model import *
from scipy import io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import PIL.Image as PIL
import numpy as np
from PIL import Image

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = full_img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

    if net.n_classes == 1:
        return (probs.cpu() > out_threshold).numpy()
    else:
        return F.one_hot(probs.cpu().argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 50).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 100 / mask.shape[0]).astype(np.uint8))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNet(1, 5, bilinear=False) 

net.to(device=device)
weight = "C:/Users/hb/Desktop/code/Influencer_learning/IL/Results/2023-04-23_20H_32M/models/participant0.pth"
net.load_state_dict(torch.load(weight, map_location=device))

index = "50"              
brain = np.load("D:/Data/BraTS2021/2D/Training/Participant1/imgs/" + index + ".npy")
mask = np.load("D:/Data/BraTS2021/2D/Training/Participant1/labels/" + index + ".npy")    

mean = np.mean(brain)
std = np.std(brain)

transform1 = transforms.Compose([
transforms.ToPILImage(),
transforms.ToTensor(),
transforms.Normalize(mean, std, inplace=False),
]) 

input = transform1(brain)

out = predict_img(net=net, full_img=input, device=device)
result = mask_to_image(mask)


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 6))
ax1.imshow(input[0])
ax1.set_title('input')
ax2.imshow(result)
ax2.set_title('prediction')
ax3.imshow(mask)
ax3.set_title('mask')
plt.savefig("C:/Users/hb/Desktop/code/Influencer_learning/FL/Results/BraTS2021/" + index + ".jpg")