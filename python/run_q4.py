import os
import numpy as np
import matplotlib
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    print(img)
    bboxes, bw = findLetters(im1)

    # plt.imshow(bw)
    clustered_bboxes = {}
    
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        if len(clustered_bboxes.keys()) == 0:
            clustered_bboxes[minr] = [(minr, minc, maxr, maxc)]
        else:
            inserted = False
            for key in clustered_bboxes:
                if abs(minr - key) < im1.shape[0]/11:
                    clustered_bboxes[key].append((minr, minc, maxr, maxc))
                    inserted = True
            if not inserted:
                clustered_bboxes[minr] = [(minr, minc, maxr, maxc)]
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    for i in clustered_bboxes:
        def takeSecond(tup):
            return tup[1]
        clustered_bboxes[i].sort(key=takeSecond)
    # plt.show()
    # plt.savefig("bbx_"+img)
    
    # find the rows using..RANSAC, counting, clustering, etc.
    sorted_bboxes_lines = []
    line_data = []
    for key in sorted(clustered_bboxes.keys()):
        sorted_bboxes_lines.append(clustered_bboxes[key])
    for sorted_bboxes in sorted_bboxes_lines:
        data = np.zeros((len(sorted_bboxes), 32*32))
        for i in range(len(sorted_bboxes)):
            # print(bbox)
            # sorted_bboxes = sorted_bboxes[i]
            bbox = sorted_bboxes[i]
            (y1,x1,y2,x2) = bbox
            (y1,x1,y2,x2) = (int(y1), int(x1), int(y2), int(x2))
            # (y1,x1,y2,x2) = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            # print(y1,x1,y2,x2)
            box_part_im = bw[y1:y2,x1:x2]
            # enlarged_im = skimage.transform.resize(, (18,18),
            #                anti_aliasing=True)
            enlarged_im = skimage.transform.resize(box_part_im, (14,14),
                        anti_aliasing=True)
            enlarged_im = skimage.filters.gaussian(enlarged_im, sigma=0.02)
            enlarged_im = skimage.transform.resize(enlarged_im, (22,22),
                        anti_aliasing=True)
            enlarged_im = np.transpose(np.pad(enlarged_im, pad_width=5,
                        mode="constant", constant_values=(np.max(enlarged_im)))**3)
            # enlarged_im = skimage.filters.gaussian(enlarged_im, sigma=0.03)
            row = np.reshape(enlarged_im, 32*32)
            data[i] = row
        line_data.append(data)
    
        
    # print(sorted_bboxes)
    # data = np.zeros((len(sorted_bboxes), 32*32))
    # for i in range(len(sorted_bboxes_L)):
    #     # print(bbox)
    #     sorted_bboxes = sorted_bboxes_L[i]
    #     bbox = sorted_bboxes[i]
    #     (y1,x1,y2,x2) = bbox
    #     (y1,x1,y2,x2) = (int(y1), int(x1), int(y2), int(x2))
    #     # (y1,x1,y2,x2) = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    #     # print(y1,x1,y2,x2)
    #     box_part_im = bw[y1:y2,x1:x2]
    #     # enlarged_im = skimage.transform.resize(, (18,18),
    #     #                anti_aliasing=True)
    #     enlarged_im = skimage.transform.resize(box_part_im, (14,14),
    #                    anti_aliasing=True)
    #     enlarged_im = skimage.filters.gaussian(enlarged_im, sigma=0.02)
    #     enlarged_im = skimage.transform.resize(enlarged_im, (22,22),
    #                    anti_aliasing=True)
    #     enlarged_im = np.transpose(np.pad(enlarged_im, pad_width=5,
    #                    mode="constant", constant_values=(np.max(enlarged_im)))**3)
    #     # enlarged_im = skimage.filters.gaussian(enlarged_im, sigma=0.03)
    #     row = np.reshape(enlarged_im, 32*32)
    #     data[i] = row
        # plt.imshow(enlarged_im)
        # plt.show()
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    res_arr = []
    for data in line_data:
        post_act = forward(data,params,'layer1')
        probs = forward(post_act,params,'output',softmax)
        # print(
        res = ""
        for i in range(probs.shape[0]):
            pred_label = np.argmax(probs[i])
            if pred_label < 26:
                res += chr(65+pred_label)
                # print(chr(65+pred_label),pred_label)
            else:
                res += chr(48+pred_label-26)
                # print(chr(48+pred_label-26),pred_label)
        res_arr.append(res)
    print(res_arr)
        
    
