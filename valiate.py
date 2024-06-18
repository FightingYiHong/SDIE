import cv2
import numpy as np
import os


def calculateIoU(Image_path,GT_path):
    input_image = cv2.imread(Image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE)

    _, input_image_binary = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)
    _, target_image_binary = cv2.threshold(target_image, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(input_image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(input_image_binary, contours, -1, (0, 0, 0), 1)
    #cv2.imshow("image", input_image_binary)

    # 计算前景和背景之间的交集和并集
    foreground_intersection = np.logical_and(input_image_binary, target_image_binary)
    foreground_union = np.logical_or(input_image_binary, target_image_binary)
    background_intersection = np.logical_and(1 - input_image_binary, 1 - target_image_binary)
    background_union = np.logical_or(1 - input_image_binary, 1 - target_image_binary)

    # 计算FBIoU
    foreground_iou = np.sum(foreground_intersection) / np.sum(foreground_union)
    background_iou = np.sum(background_intersection) / np.sum(background_union)
    fbiou = 0.5 * (foreground_iou + background_iou)




    # intersection = np.logical_and(input_image_binary, target_image_binary)
    # union = np.logical_or(input_image_binary, target_image_binary)
    #
    # iou = np.sum(intersection) / np.sum(union)
    return fbiou

def get_Image_list(path):
    Image_list=[]
    for file_name in os.listdir(path):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            Image_list.append(os.path.join(path,file_name))
    return sorted(Image_list)



Image_path="Processed/Images"
GT_path="Processed/GT"
Image_list=get_Image_list(Image_path)
GT_list=get_Image_list(GT_path)
total_IoU=0
for i in range (len(Image_list)):
    total_IoU+=calculateIoU(Image_list[i],GT_list[i])
fbmIoU=total_IoU/len(Image_list)

print("fbmIoU:",fbmIoU)




