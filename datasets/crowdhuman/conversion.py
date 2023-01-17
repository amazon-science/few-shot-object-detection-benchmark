# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import pandas as pd
from pycocotools.coco import COCO
from collections import OrderedDict

annotations_headers = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax']
extras_headers = ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
all_headers = annotations_headers + extras_headers


def write_annotations_to_openimages_csv(annotations, path='annotations.csv', extras={}):
    if annotations:
        for header in annotations_headers:
            assert header in annotations
        df = pd.DataFrame(annotations)
        if extras:
            for header in extras_headers:
                assert header in extras
        else:
            anno_len = len(annotations['ImageID'])
            extras = OrderedDict()
            for header in extras_headers:
                extras[header] = [-1] * anno_len
        df_extra = pd.DataFrame(extras)
        df = pd.concat([df, df_extra], axis=1)

        df.to_csv(path, index=False, float_format='%.3f')
    else:
        print("Annotation is empty")


class OpenImagesWriterFromCOCO:
    def __init__(self, train_json_file, test_json_file, output_dir):
        self.train_json_file = train_json_file
        self.test_json_file = test_json_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.classes = {}

    def write_class_map(self):
        path = os.path.join(self.output_dir, 'class_map.csv')
        with open(path, 'w+') as fp:
            for cur_cls in self.classes.keys():
                fp.write(str(self.classes[cur_cls]) + '\t' + str(cur_cls) + '\n')

    def convert_files(self):


        for cur_set in ['train', 'test']:
            path = os.path.join(self.output_dir, cur_set + '.csv')
            images = []

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            bbox_classes = []

            if cur_set == 'train':
                coco = COCO(self.train_json_file)
            else:
                if self.test_json_file:
                    coco = COCO(self.test_json_file)
                else:
                    continue
            imgIds = coco.getImgIds()
            for i, imgId in enumerate(imgIds):
                if i % 1000 == 0:
                    print(i, len(imgIds))
                img = coco.loadImgs(imgId)[0]
                ann_ids = coco.getAnnIds(imgIds=[imgId])
                objs = coco.loadAnns(ann_ids)


                for obj in objs:
                    cat = coco.loadCats(obj['category_id'])[0]
                    image_name = img['file_name']
                    images.append(image_name)
                    bbox_classes.append(cat['name'])
                    x, y, w, h = obj['bbox']
                    xmins.append(x)
                    ymins.append(y)
                    xmaxs.append(x + w - 1)
                    ymaxs.append(y + h - 1)
            box_class_id = []
            for box_class in bbox_classes:
                try:
                    cur_class = self.classes[box_class]
                except KeyError:
                    if cur_set == 'test':
                        print('warning: testing class '+str(cur_class) + ' not in training classes')
                        raise AssertionError
                    self.classes[box_class] = len(self.classes.keys())
                    cur_class = self.classes[box_class]
                box_class_id.append(cur_class)
            annotations = OrderedDict([('ImageID', images), ('Source', ['freeform'] * len(images)),
                                       ('LabelName', box_class_id), ('Confidence', [1.0] * len(images)),
                                       ('XMin', [xmin for xmin in xmins]), ('XMax', [xmax for xmax in xmaxs]),
                                       ('YMin', [ymin for ymin in ymins]), ('YMax', [ymax for ymax in ymaxs])])
            write_annotations_to_openimages_csv(annotations, path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert annotation in COCO JSON format to OpenImages')
    parser.add_argument('--train_json_file', type=str, help='JSON file contains COCO format data')
    parser.add_argument('--test_json_file', type=str, help='JSON file contains COCO format data', default='')
    parser.add_argument('--output_dir', type=str, default=None, help='Output oi_fles folder')
    args = parser.parse_args()

    converter = OpenImagesWriterFromCOCO(args.train_json_file, args.test_json_file, args.output_dir)
    converter.convert_files()
    converter.write_class_map()

