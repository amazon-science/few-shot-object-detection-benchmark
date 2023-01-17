# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import argparse

'''
Note 1: "ignored regions" and "others" classes and annotations in these classes are removed
Note 2: original dataset has 3 occlusion classes:
no occlusion = 0 (occlusion ratio 0%)
partial occlusion = 1 (occlusion ratio 1% ~ 50%)
heavy occlusion = 2 (occlusion ratio 50% ~ 100%)
we take IsOccluded = (occlusion > 0)
'''
parser = argparse.ArgumentParser(description='Convert annotation to OpenImages')
parser.add_argument('--root', type=str, help='./datasets/visdrone2019/')
args = parser.parse_args()

root = args.root

oi_anno_dir = os.path.join(root, 'OI_Annotations')
if not os.path.isdir(oi_anno_dir):
    os.makedirs(oi_anno_dir)
oi_anno = os.path.join(oi_anno_dir, '{}.csv')
oi_class_map = os.path.join(oi_anno_dir, 'class_map.csv')
head_str = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'

# class_names = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
#                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
# ignore 0 and 11; label <- label-1
class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
with open(oi_class_map, 'w') as f:
    for i, class_name in enumerate(class_names):
        write_me = f'{i}\t{class_name}\n'
        f.write(write_me)

types = ['train', 'val', 'test-dev']
for type in types:
    image_dir = os.path.join(root, f'VisDrone2019-DET-{type}/images')
    anno_dir = os.path.join(root, f'VisDrone2019-DET-{type}/annotations')

    file_names = sorted(os.listdir(image_dir))
    if type == 'train':
        out, mode = oi_anno.format('train'), 'w'
    elif type == 'val':
        out, mode = oi_anno.format('train'), 'a'
    elif type == 'test-dev':
        out, mode = oi_anno.format('test'), 'w'
    print(type)
    with open(out, mode) as f:
        if mode == 'w':
            f.write(head_str)
        for i, file_name in enumerate(file_names):
            if i % 100 == 0:
                print(f'{i:4d} / {len(file_names):4d}')
            if type == 'test-dev':
                image_path = os.path.join(f'test2019', file_name)
            else:
                image_path = os.path.join(f'{type}2019', file_name)
            anno_path = os.path.join(anno_dir, os.path.splitext(file_name)[0] + '.txt')
            anno = open(anno_path, 'r').readlines()
            for line in anno:
                try:
                    x1, y1, w, h, score, label, trun, occl = line.strip().split(',')[:8]
                except ValueError:
                    print(line.strip().split(','))
                    print(line)
                    assert False, 'parse error'
                if (label == '0') or (label == '11'):
                    continue
                write_me = f'{image_path},freeform,{int(label)-1},{score},{int(x1)},{int(x1)+int(w)-1},{int(y1)},{int(y1)+int(h)-1},{int(int(occl) > 0)},{trun},-1,-1,-1\n'
                f.write(write_me)

    '''
     <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>


        Name                                                  Description
    -------------------------------------------------------------------------------------------------------------------------------
     <bbox_left>         The x coordinate of the top-left corner of the predicted bounding box

     <bbox_top>          The y coordinate of the top-left corner of the predicted object bounding box

     <bbox_width>        The width in pixels of the predicted object bounding box

    <bbox_height>        The height in pixels of the predicted object bounding box

       <score>           The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing
                         an object instance.
                         The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation,
                         while 0 indicates the bounding box will be ignored.

    <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1),
                         people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10),
                         others(11))

    <truncation>         The score in the DETECTION result file should be set to the constant -1.
                         The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame
                         (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).

    <occlusion>          The score in the DETECTION file should be set to the constant -1.
                         The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0
                         (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2
                         (occlusion ratio 50% ~ 100%)).
    '''
    # oi anno example:
    # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
    # val2017/000000532481.jpg,freeform,2,1.000,250.820,319.930,168.260,232.140,-1,-1,-1,-1,-1

    # anno example: Annotations/017164.jpg.txt
    # 684,8,273,116,0,0,0,0
    # 406,119,265,70,0,0,0,0
    # 255,22,119,128,0,0,0,0
    # 1,3,209,78,0,0,0,0
    # 708,471,74,33,1,4,0,1
    # 639,425,61,46,1,4,0,0
    # 594,399,64,51,1,4,0,0
    # 562,390,61,38,1,4,0,0
    # 540,372,65,33,1,4,0,1