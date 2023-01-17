# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert annotation to OpenImages')
parser.add_argument('--root', type=str, help='./datasets/oktoberfest/')
args = parser.parse_args()

root = args.root

image_dir = root
anno_dir = root

oi_anno_dir = os.path.join(root, 'OI_Annotations')
if not os.path.isdir(oi_anno_dir):
    os.makedirs(oi_anno_dir)
oi_anno = os.path.join(oi_anno_dir, '{}.csv')
oi_class_map = os.path.join(oi_anno_dir, 'class_map.csv')
head_str = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n'

class_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']
with open(oi_class_map, 'w') as f:
    for i, class_name in enumerate(class_names):
        write_me = f'{i}\t{class_name}\n'
        f.write(write_me)

types = ['train', 'test']
for type in types:
    anno_path = os.path.join(anno_dir, f'{type}_files.txt')
    anno = open(anno_path, 'r').readlines()
    out, mode = oi_anno.format(type), 'w'
    with open(out, mode) as f:
        if mode == 'w':
            f.write(head_str)
        for l in anno:
            s = l.split(' ')
            image_path = os.path.join(type, s[0])
            for i in range(int(s[1])):
                c, x, y, w, h = [int(x) for x in s[2 + 5 * i:7 + 5 * i]]
                write_me = f'{image_path},freeform,{c},1.0,{x},{x+w},{y},{y+h},-1,-1,-1,-1,-1\n'
                f.write(write_me)

# class names and coordinate transform https://github.com/a1302z/OktoberfestFoodDataset/blob/master/ShowAnnotations.py

# anno example:
# 1526742310765_20.jpg 6 11 1008 570 86 74 11 924 590 85 80 11 1045 658 89 82 11 956 672 84 80 1 516 350 265 164 1 568 468 269 182
# 1526744530681_20.jpg 6 1 1100 496 216 212 1 940 456 228 200 0 624 416 124 140 0 720 368 200 140 0 808 436 196 140 0 796 576 152 120

# oi anno example:
# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
# val2017/000000532481.jpg,freeform,2,1.000,250.820,319.930,168.260,232.140,-1,-1,-1,-1,-1