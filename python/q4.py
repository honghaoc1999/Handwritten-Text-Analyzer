import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    sigma_est = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    print(sigma_est)
    denoised_image = skimage.restoration.denoise_wavelet(image, multichannel=True, rescale_sigma=True)
    # plt.imshow(denoised_image,cmap="gray")
    # plt.savefig("test_save.png")
    # plt.show()
    grayscale_image = skimage.color.rgb2gray(denoised_image)
    thresh = skimage.filters.threshold_otsu(grayscale_image)
    binary_image = grayscale_image < thresh
    # plt.imshow(~binary_image/np.max(binary_image),cmap="gray")
    bw = ~binary_image/np.max(binary_image)
    # plt.savefig("test_binary.png")
    # plt.show()
    
    morph_image = skimage.morphology.binary.binary_opening(binary_image)
    
    (labeled_image, return_num) = skimage.measure.label(morph_image, connectivity=None, background=0, return_num=True)
    print(return_num)
    # plt.imshow(labeled_image,cmap="gray")
    # plt.savefig("test_save.png")
    # plt.show()
    
    for i in range(return_num):
        indices = np.argwhere(labeled_image == i+1)
        # print(indices.shape[0])
        # if indices.shape[0] < 100:
        #     continue
        # print(len(indices))
        left_top_corner = np.amin(indices, axis=0)
        right_bottom_corner = np.amax(indices, axis=0)
        # print(left_top_corner, right_bottom_corner)
        boxes = np.zeros(4)
        boxes[:2] = left_top_corner
        boxes[2:] = right_bottom_corner
        if indices.shape[0] > 300 and (right_bottom_corner[0] - left_top_corner[0]) > image.shape[0]/40:
            bboxes.append(boxes)
        
    # bw = labeled_image/np.max(labeled_image)
        
    return bboxes, bw
# import os
# import numpy as np
# import matplotlib
# # matplotlib.use('agg') 
# import matplotlib.pyplot as plt
# import matplotlib.patches
# 
# import skimage
# import skimage.measure
# import skimage.color
# import skimage.restoration
# import skimage.io
# import skimage.filters
# import skimage.morphology
# import skimage.segmentation
# im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',"03_haiku.jpg")))
# bboxes, bw = findLetters(im1)
# plt.imshow(bw)
# for bbox in bboxes:
#     minr, minc, maxr, maxc = bbox
#     rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                             fill=False, edgecolor='red', linewidth=2)
#     plt.gca().add_patch(rect)
# plt.show()
# # plt.show()
# # import cv2 as cv
# # image = cv.imread("../images/02_letters.jpg")
# # findLetters(image)
# import scipy.io
# train_data = scipy.io.loadmat('../data/nist36_train.mat')
# print(train_data['train_labels'][-1],len(train_data['train_labels'][-1]))
# plt.imshow(np.reshape(train_data['train_data'][1500],(32,32)))
# plt.show()
# print(chr(65))
# i= 0
# 
# for right in range(4,225,4):
#     i+=1
# print(i, len(range(4,225)))
# arr = np.zeros((2,2))
# print(np.pad(arr, pad_width=((4,4),(2,2)), mode="constant",constant_values=(1)))
# mat = np.array([[0,1,2],[2,3,4]])
# print(np.reshape(mat, 