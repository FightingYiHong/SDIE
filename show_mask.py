import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import matplotlib
def reserve_result(anns,file_name):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    img*=255
    img=img.astype(np.uint8)
    img=Image.fromarray(img)
    save_path=os.path.join('mask',file_name)
    img=img.convert('RGB')
    img.save(save_path)
    print("successful save")
    # img=img*255
    # img=img.astype(np.uint8)
    # img=Image.fromarray(img)
    # save_path=('output')
    # os.makedirs(save_path,exist_ok=True)
    # img.save(os.path.join(save_path, file_name))




def get_image_files(folder):
    image_files=[]
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.jpg','.jpeg','.png','gif')):
            image_files.append(os.path.join(folder,file_name))
            print(file_name)
    return image_files


folder='FSSD-12/Steel_Ri/Images'
files=get_image_files(folder)
for file_name in files:
    image=cv2.imread(file_name)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    sam_checkpoint = "/data/yhliu/segment-anything/check/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    reserve_result(masks,file_name)


