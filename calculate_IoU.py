import os
import cv2
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt

def reserve_result(anns,Image_path):
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
    img = img.convert('RGB')
    img.save(Image_path)


def get_mask(Image):
    image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "/data/yhliu/segment-anything/check/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    reserve_result(masks,'masks.jpg')



def bucket_sort(image):
    height, width = image.shape[:2]
    gray_counts = {}
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if pixel in gray_counts:
                gray_counts[pixel] += 1
            else:
                gray_counts[pixel] = 1

    most_common_gray = max(gray_counts, key=gray_counts.get)


    result_image = np.where(abs(image - most_common_gray)<10, 0, 255).astype(np.uint8)
    return result_image

def calculateIoU(input_image,target_image):




    _, input_image_binary = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)
    _, target_image_binary = cv2.threshold(target_image, 127, 255, cv2.THRESH_BINARY)
    # print(input_image_binary)
    # print()
    # print(target_image_binary)

    contours, hierarchy = cv2.findContours(input_image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(input_image_binary, contours, -1, (0, 0, 0), 1)

    foreground_intersection = np.logical_and(input_image_binary, target_image_binary)
    foreground_union = np.logical_or(input_image_binary, target_image_binary)
    background_intersection = np.logical_and(1 - input_image_binary, 1 - target_image_binary)
    background_union = np.logical_or(1 - input_image_binary, 1 - target_image_binary)

    foreground_iou = np.sum(foreground_intersection) / np.sum(foreground_union)
    background_iou = np.sum(background_intersection) / np.sum(background_union)
    fbmiou = 0.5 * (foreground_iou + background_iou)




    intersection = np.logical_and(input_image_binary, target_image_binary)
    union = np.logical_or(input_image_binary, target_image_binary)
    iou = np.sum(intersection) / np.sum(union)
    return iou,fbmiou




def SAM(Image,GT):
    get_mask(Image)
    image = cv2.imread('masks.jpg')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = bucket_sort(gray_image)


    kernel = np.ones((10, 10), np.uint8)
    opened_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)

    cv2.imwrite('opened_pic.jpg',opened_image)
    input_image = cv2.imread('opened_pic.jpg', cv2.IMREAD_GRAYSCALE)
    mIoU,fbmIoU=calculateIoU(input_image,GT)
    return fbmIoU


