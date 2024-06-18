import numpy as np
import cv2
# import math
from skimage.color import rgb2lab, lab2rgb
from scipy.sparse import spdiags, linalg

def bi_wls_filter(luma, lambda_, alpha):
    """
    edge-preserving smoothing via weighted least squares (WLS)
        u = F_λ (g) = (I + λ L_g)^(-1) g
        L_g = D_x^T A_x D_x +D_y^T A_y D_y

    arguments:
        luma (2-dim array, required) - the input image luma
        lambda_ (float) - balance between the data term and
            the smoothness term
        alpha (float) - a degree of control over the affinities
            by non-lineary scaling the gradients

    return:
        out (2-dim array)
    """
    EPS = 1e-4
    DIM_X = 1
    DIM_Y = 0
    height, width = luma.shape[0: 2]
    size = height * width
    log_luma = np.log(luma + EPS)

    # affinities between adjacent pixels based on gradients of luma
    # dy
    diff_log_luma_y = np.diff(a=log_luma, n=1, axis=DIM_Y)
    diff_log_luma_y = - lambda_ / (np.abs(diff_log_luma_y) ** alpha + EPS)
    diff_log_luma_y = np.pad(
        array=diff_log_luma_y, pad_width=((0, 1), (0, 0)),
        mode="constant"
    )
    diff_log_luma_y = diff_log_luma_y.ravel()

    # dx
    diff_log_luma_x = np.diff(a=log_luma, n=1, axis=DIM_X)
    diff_log_luma_x = - lambda_ / (np.abs(diff_log_luma_x) ** alpha + EPS)
    diff_log_luma_x = np.pad(
        array=diff_log_luma_x, pad_width=((0, 0), (0, 1)),
        mode="constant"
    )
    diff_log_luma_x = diff_log_luma_x.ravel()

    # construct a five-point spatially inhomogeneous Laplacian matrix
    diff_log_luma = np.vstack((diff_log_luma_y, diff_log_luma_x))
    smooth_weights = spdiags(data=diff_log_luma, diags=[-width, -1],
                             m=size, n=size)

    w = np.pad(array=diff_log_luma_y, pad_width=(width, 0), mode="constant")
    w = w[: -width]
    n = np.pad(array=diff_log_luma_x, pad_width=(1, 0), mode="constant")
    n = n[: -1]

    diag_data = 1 - (diff_log_luma_x + w + diff_log_luma_y + n)
    smooth_weights = smooth_weights + smooth_weights.transpose() + \
                     spdiags(data=diag_data, diags=0, m=size, n=size)

    out, _ = linalg.cg(A=smooth_weights, b=luma.ravel())
    out = out.reshape((height, width))
    # out = np.clip(a=out, a_min=0, a_max=100)

    return out

def wls_filter(img,lambda_, alpha):
    lab = rgb2lab(img)
    luma = lab[:, :, 0]
    res = bi_wls_filter(luma=luma, lambda_=lambda_, alpha=alpha)
    image_out = lab2rgb(
        lab=np.asarray([res, lab[..., 1], lab[..., 2]]).transpose(1, 2, 0)
    )
    image_out_uint8 = np.clip(image_out * 255, 0, 255).astype('uint8')

    # 使用cv2.cvtColor将RGB图像转换为灰度图像
    gray_image = cv2.cvtColor(image_out_uint8, cv2.COLOR_RGB2GRAY)
    return gray_image

def bilateral_filter(image, diameter=15, sigma_color=75, sigma_space=75):
    """
    Bilateral Filter
    """
    return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

def imsharp_filter(image, amount=0.5, filter_size=5):
    """
    Image Sharpening Filter
    """
    blurred = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def guided_filter(image, radius=5, epsilon=0.1):
    """
    Guided Filter
    """
    # Convert image to float32 type for guidedFilter
    img_float = np.float32(image) / 255.0
    guided = cv2.ximgproc.guidedFilter(guide=img_float, src=img_float, radius=radius, eps=epsilon)
    # Convert back to original data type
    guided = np.uint8(guided * 255)
    return guided

def histogram_equalization(image,filter_size=5):
    """
    Histogram Equalization
    """

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(filter_size, filter_size))
    img_output = clahe.apply(image)
    return img_output

# Example of loading an image and applying these filters
action=[0.15,0.15,0.15,0.15,0.15,0,1.5,2]
def gamma(image,ga=1.0):
    invga=1.0/ga
    table=np.array([((i/255.0)**invga)*255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image,table)
def mix_filters(image,action=[0.15,0.15,0.15,0.15,0.15,0,1.5,0.125,1.2],filter_size=5):

    first_six = action[:6]
    for i in range(len(action)):
            action[i]+=1
    s=sum(first_six)
    action[6] = 1.5
    action[7] = 0.125
    action[8] = 1.2
    # 归一化前六项
    action[:6] = [x / s for x in first_six]
    pseudo_rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    wls_filtered_image = wls_filter(pseudo_rgb_image,lambda_=action[7],alpha=action[8])
    bilateral_filtered_image = bilateral_filter(image)
    imsharp_filtered_image = imsharp_filter(image,amount=action[6],filter_size=filter_size)
    guided_filtered_image = guided_filter(image,radius=filter_size)
    histogram_equalized_image = histogram_equalization(image,filter_size=filter_size)
    difference=[]
    images = [
        wls_filtered_image,
        bilateral_filtered_image,
        imsharp_filtered_image,
        guided_filtered_image,
        histogram_equalized_image
    ]
    #
    #
    #
    # for i in images:
    #     difference.append(np.sqrt(np.sum(np.square(image.astype("float") - i.astype("float")))))
    # for i in range(len(images)):
    #     action[i]=difference[i]/sum(difference)
    # first_six = action[:5]
    # s = sum(first_six)
    #
    # # 归一化前六项
    # action[:5] = [x for x in first_six]
    # action[5]=0
    # print(action)
    weighted_sum_image = None
    for i in range(5):
        image=images[i]
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        image = image.astype(np.float32)
        if weighted_sum_image is None:
            weighted_sum_image = np.zeros_like(image)
        weighted_sum_image += image * action[i]
    weighted_sum_image += image * action[5]
    weighted_sum_image = np.clip(weighted_sum_image, 0, 255).astype('uint8')
    return weighted_sum_image


