# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 22:37:02 2015
@author: shang
"""

import cv2
import json
import numpy as np
import itertools
import matplotlib.pyplot as plt
# matplotlib inline
from scipy.misc import imread
from IPython import display
import os
import sys
sys.path.append('..')
import apollocaffe # Make sure that caffe is on the python path:

from utils.annolist import AnnotationLib as al
from crowd_train import (forward, get_numbers)
from utils import (annotation_jitter, image_to_h5,
                   annotation_to_h5, load_data_mean, Rect, stitch_rects)
from utils.annolist import AnnotationLib as al

def load_idl(idlfile, data_mean, net_config, jitter=True):
    """Take the idlfile, data mean and net configuration and create a generator
    that outputs a jittered version of a random image from the annolist
    that is mean corrected."""

    annolist = al.parse(idlfile)
    annos = [x for x in annolist]
    for anno in annos:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(idlfile)), anno.imageName)
    while True:
        for anno in annos:
            if jitter:
                jit_image, jit_anno = annotation_jitter(
                    anno, target_width=net_config["img_width"],
                    target_height=net_config["img_height"])
            else:
                jit_image = imread(anno.imageName)
                jit_anno = anno
            image = image_to_h5(jit_image, data_mean, image_scaling=1.0)
            boxes, box_flags = annotation_to_h5(
                jit_anno, net_config["grid_width"], net_config["grid_height"],
                net_config["region_size"], net_config["max_len"])
            # numbers = get_numbers(
            #     jit_anno, net_config["grid_width"], net_config["grid_height"],
            #     net_config["region_size"])
            yield {"imname": anno.imageName, "raw": jit_image, "image": image,
                   "boxes": boxes, "box_flags": box_flags, "rects": anno.rects}
                   
def stitch_nums(nums, net_config):
    region_size = net_config["region_size"]
    ref_width = net_config["img_width"] / net_config["grid_width"]
    ref_height = net_config["img_height"] / net_config["grid_height"]
    assert(region_size >= ref_width and region_size >= ref_height)
#    cell_rects = [[Rect(x*ref_width, y*ref_height, (x+1)*ref_width-1, (y+1)*ref_height-1, 0) 
#                for x in range(net_config["grid_width"])] 
#                for y in range(net_config["grid_height"])]
    acc_nums = np.zeros((net_config["grid_height"], net_config["grid_width"]))
#    acc_nums = [[0 for x in range(net_config["grid_width"])] 
#                for y in range(net_config["grid_height"])]
#    for i in range(net_config["grid_height"]):
#        for j in range(net_config["grid_width"]):
#            r = cell_rects[i][j]
#            cx = 0.5 * (r.x1 + r.x2)
#            cy = 0.5 * (r.y1 + r.y2)
#            ext = 0.5 * (region_size - 1)
#            cell_rects[i][j].x1 = round(max(0, cx - ext))
#            cell_rects[i][j].x2 = round(min(net_config["img_width"]-1, cx + ext))
#            cell_rects[i][j].y1 = round(max(0, cy - ext))
#            cell_rects[i][j].y2 = round(min(net_config["img_height"]-1, cx + ext))
#            cell_rects[i][j].score = nums[i * net_config["grid_width"] + j]

    ratio = 1.0 * ref_width / region_size
    for i in range(net_config["grid_height"]):
        for j in range(net_config["grid_width"]):
            neighbors = []
            if i > 0:
                neighbors.append(nums[(i-1)*net_config["grid_width"] + j])
            if j > 0:
                neighbors.append(nums[i*net_config["grid_width"] + j-1])
            if i < net_config["grid_height"]-1:
                neighbors.append(nums[(i+1)*net_config["grid_width"] + j])
            if j < net_config["grid_width"]-1:
                neighbors.append(nums[i*net_config["grid_width"] + j+1])
            acc_nums[i][j] = nums[i * net_config["grid_width"] + j] * (1-len(neighbors)*ratio) + sum(neighbors) * ratio
    return acc_nums
    
            

def test(config):
    """ Takes the config, run test program
    """

    data_mean = load_data_mean(config["data"]["idl_mean"],
                               config["net"]["img_width"],
                               config["net"]["img_height"], image_scaling=1.0)

    num_test_images = 5

    # Warning: load_idl returns an infinite generator. Calling list() before islice() will hang.
    test_list = list(itertools.islice(
            load_idl(config["data"]["test_idl"], data_mean, config["net"], False),
            0,
            num_test_images))
    img = np.copy(test_list[-1]["raw"])
    # plt.imshow(img)

    net = apollocaffe.ApolloNet()
    net.phase = 'test'
    forward(net, test_list[0], config["net"], True)
    #net.load("data/snapshot/reinspect_hcs_600000.h5")
    net.load("data/reinspect_800000.h5")
    
    annolist = al.AnnoList()
    net_config = config["net"]
    pix_per_w = net_config["img_width"]/net_config["grid_width"]
    pix_per_h = net_config["img_height"]/net_config["grid_height"]

    if config.has_key("conf_th"):
        conf_th = config["conf_th"]
    else:
        conf_th = 0.6

    mae = 0.
    for i in range(num_test_images):
        inputs = test_list[i]
        bbox_list, conf_list, num = forward(net, inputs, net_config, True)

        img = np.copy(inputs["raw"])
        # handling all rects
        all_rects = [[[] for x in range(net_config["grid_width"])] for y in range(net_config["grid_height"])]
        for n in range(len(bbox_list)):
            for k in range(net_config["grid_height"] * net_config["grid_width"]):
                y = int(k / net_config["grid_width"])
                x = int(k % net_config["grid_width"])
                bbox = bbox_list[n][k]
                conf = conf_list[n][k,1].flatten()[0]
                # notice the output rect [cx, cy, w, h]
                # cx means center x-cord
                abs_cx = pix_per_w/2 + pix_per_w*x + int(bbox[0,0,0])
                abs_cy = pix_per_h/2 + pix_per_h*y + int(bbox[1,0,0])
                w = bbox[2,0,0]
                h = bbox[3,0,0]
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))
        acc_rects = stitch_rects(all_rects)
        
        num = num.flatten()
        acc_num = stitch_nums(num, net_config)
        
            
        display = True
        if display:
            for rect in acc_rects:
                if rect.true_confidence < conf_th:
                    continue
                cv2.rectangle(img,
                              (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                              (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                              (255,0,0),
                              2)
            img_name = './data/tmp/%05d.jpg' % i
            plt.imsave(img_name, img)
            plt.figure(figsize=(15,10))
            plt.imshow(img)

        anno = al.Annotation()
        anno.imageName = inputs["imname"]
        number = 0
        for rect in acc_rects:
            r = al.AnnoRect()
            r.x1 = rect.cx - rect.width/2.
            r.x2 = rect.cx + rect.width/2.
            r.y1 = rect.cy - rect.height/2.
            r.y2 = rect.cy + rect.height/2.
            r.score = rect.true_confidence
            anno.rects.append(r)
            if r.score > conf_th:
                number += 1
        annolist.append(anno)
        
        # handle count
        mae += abs(number - len(inputs["rects"]))
        # number = round(sum(num))
        print anno.imageName, number, len(inputs["rects"]), abs(number - len(inputs["rects"]))
    print mae / num_test_images

def main():
    """Sets up all the configurations for apollocaffe, and ReInspect
    and runs the test."""
    parser = apollocaffe.base_parser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    print ("Test config file is " + config["data"]["test_idl"])
    apollocaffe.set_random_seed(config["solver"]["random_seed"])
    apollocaffe.set_device(0) # gpu
    test(config)

if __name__ == "__main__":
    main()
