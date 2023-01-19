# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import numpy as np
import json
import csv
import pickle
import cv2
import time
from PIL import Image
from detectron2.structures import BoxMode
from collections import defaultdict, OrderedDict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

class openimages(object):
    def _init__(self, data_dir, dataset_name, cache_path, sampling_data_dir=None, post_fix_key=''):
        # name, paths
        self._dataset_name = dataset_name
        self._data_path = osp.join(data_dir, dataset_name)
        self._class_map_file = osp.join(data_dir, dataset_name, 'OI_Annotations', 'class_map.csv')
        self._image_path = osp.join(data_dir, dataset_name, 'JPEGImages')
        self._cache_path = cache_path
        self._data_dir = data_dir
        if sampling_data_dir is not None:
            self._data_dir = sampling_data_dir
            self._class_map_file = osp.join(sampling_data_dir, dataset_name, 'OI_Annotations', 'class_map.csv')
        # class name
        self._classes = self.get_class_names(self._class_map_file)
        self._num_classes = len(self._classes)
        # class name to ind    (0~num_classes-1)
        self.image_set_index = []
        self.cat_data = defaultdict(list)
        self.dataset_size = None

        for d in ["train", "test"]:
            DatasetCatalog.register(dataset_name + '_' + d + post_fix_key, lambda d=d: self.gt_roidb(d))
            MetadataCatalog.get(dataset_name + '_' + d + post_fix_key).set(thing_classes=self._classes)
            MetadataCatalog.get(dataset_name + '_' + d + post_fix_key).set(evaluator_type='coco')
        self.metadata = MetadataCatalog.get(dataset_name + '_train' + post_fix_key)

    def get_class_names(self, path):
        classes = []
        int_classes = {}
        with open(path, 'r', encoding="utf-8") as fp:
            line = fp.readline()
            while line:
                processed_line = line.rstrip().split('\t')
                classes.append(processed_line[1])
                int_classes[int(processed_line[0])] = processed_line[1]
                line = fp.readline()
        classes = [int_classes[class_id] for class_id in sorted(int_classes.keys())]
        return classes

    def num_classes(self):
        return self._num_classes

    def get_dataset_size(self, image_set):
        _csv_file = osp.join(self._data_dir, self._dataset_name, 'OI_Annotations', image_set + '.csv')

        image_name = set()
        f = open(_csv_file, 'r')
        for i, lines in enumerate(f):
            if i == 0:
                continue
            words = lines.split(',')
            if int(words[2]) not in range(self._num_classes):
                print(words[2])
                continue
            imid = words[0]
            image_name.add(imid)

        f.close()

        return len(image_name)

    def get_image_size(self, csv_file, image_set):
        """
        Get image height and width for Openimage dataset
        :return: dict of image_ind to (height, width)
        """
        pickle_file = os.path.join(self._cache_path, self._dataset_name+'_imid_to_size_' + image_set + '.pkl')
        print(pickle_file)
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                imid_to_size = pickle.load(f)
        else:
            imid_to_size = {}
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for i, r in enumerate(reader):
                    if i % 100000 == 0:
                        print(i)
                    img_path = os.path.join(self._image_path, r[0])
                    if img_path not in imid_to_size:
                        im = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                        if im is None:
                            print(img_path)
                        else:
                            height, width, channels = im.shape
                            imid_to_size[img_path] = (height, width)
            # save to pickle file
            with open(pickle_file, 'wb') as fp:
                pickle.dump(imid_to_size, fp)
        return imid_to_size

    def gt_roidb(self, image_set):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        _name = self._dataset_name + '_' + image_set
        _csv_file = osp.join(self._data_dir, self._dataset_name, 'OI_Annotations', image_set + '.csv')
        cache_file = osp.join(self._cache_path, _name + '_dict.pkl')

        imid_to_size = self.get_image_size(_csv_file, image_set)

        f = open(_csv_file, 'r')

        data_map = OrderedDict()
        for i, lines in enumerate(f):
            if i == 0:
                continue
            ## ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
            words = lines.split(',')
            if int(words[2]) not in range(self._num_classes):
                print(words[2])
                continue
            imid = words[0]
            img_path = os.path.join(self._image_path, imid)
            classid = int(words[2])
            try:
                height, width = imid_to_size[img_path]
            except KeyError:
                im = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
                height, width, channels = im.shape
            x1 = float(words[4])
            x2 = float(words[5])
            y1 = float(words[6])
            y2 = float(words[7])
            if not (x2 > x1 and y2 > y1):
                print(x1, x2, y1, y2)
                continue
            crowd = int(words[10])
            if crowd > 0:
                continue
            if imid not in data_map:
                data_map[imid] = {}
                data_map[imid]['image'] = img_path
                data_map[imid]['height'] = height
                data_map[imid]['width'] = width
                data_map[imid]['gt_classes'] = []
                data_map[imid]['boxes'] = []

            cls_id = int(classid)
            data_map[imid]['gt_classes'].append(cls_id)
            data_map[imid]['boxes'].append([x1, y1, x2, y2])

        f.close()

        dataset_dicts = []
        keys = data_map.keys()
        for idx, key in enumerate(keys):
            record = {}
            boxes = np.array(data_map[key]['boxes'])
            gt_classes = np.array(data_map[key]['gt_classes'])
            if len(boxes) == 0:
                continue

            record["file_name"] = data_map[key]['image']
            record["image_id"] = idx
            record["height"] = int(data_map[key]['height'])
            record["width"] = int(data_map[key]['width'])

            objs = []
            for ii, obj in enumerate(data_map[key]['boxes']):
                obj = {
                    "bbox": [obj[0], obj[1], obj[2], obj[3]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(gt_classes[ii]),
                    "iscrowd": int(0)
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)

        print('wrote dataset_dicts to {}'.format(cache_file))
        self.dataset_size = len(dataset_dicts)
        return dataset_dicts
