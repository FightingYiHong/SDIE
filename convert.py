import cv2
import numpy as np
import os
def get_image_files(folder):
    image_files = []
    for i in os.listdir(folder):
        path=folder+'/'+i
        for j in os.listdir(path):
            if j.lower().endswith(('.jpg', '.jpeg', '.png', 'gif')):
                image_files.append(os.path.join(path, j))
    print(image_files)
    return image_files

def bucket_sort(image):
    # 遍历每个像素点，统计灰度值出现的次数
    print(image.shape)
    height, width = image.shape[:2]
    gray_counts = {}
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            if pixel in gray_counts:
                gray_counts[pixel] += 1
            else:
                gray_counts[pixel] = 1

    # 找到灰度占比最多的灰度值
    most_common_gray = max(gray_counts, key=gray_counts.get)

    # 将图像进行二值化处理
    result_image = np.where(abs(image - most_common_gray)<2, 0, 255).astype(np.uint8)
    return result_image


folder='mask/FSSD-12'
source_path='Processed'
files=get_image_files(folder)
for file in files:
    image=cv2.imread(file)
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = bucket_sort(gray_image)
    # 进行开操作
    kernel = np.ones((5, 5), np.uint8)
    opened_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
    # 保存处理后的图片
    base_path=os.path.basename(file)
    save_path=os.path.join(source_path,base_path)

    cv2.imwrite(save_path,opened_image)





