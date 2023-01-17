# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from pascal_voc_io import PascalVocReader
import os
from collections import OrderedDict
import argparse
import pandas as pd

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

class OpenImagesWriterFromVOC:
    def __init__(self):
        self.file_list = []
        self.reader = PascalVocReader()
        self.boxes = []
        self.classes = OrderedDict()

    def reset(self):
        self.file_list = []
        self.boxes = []
        self.reader.reset()

    def read_xmls(self, file_list, xml_folder):
        assert os.path.exists(file_list)
        with open(file_list, 'r') as fp:
            line = fp.readline()
            while line:
                line = line.rstrip()
                if not line.endswith('.xml'):
                    line = line + '.xml'
                self.file_list.append(os.path.join(xml_folder, line))
                line = fp.readline()

    def read_xmls_with_exceptions(self, file1, file2, xml_folder):
        assert os.path.exists(file1)
        assert os.path.exists(file2)
        with open(file1, 'r') as fp:
            line = fp.readline()
            while line:
                line = line.rstrip()
                if not line.endswith('.xml'):
                    line = line + '.xml'
                self.file_list.append(os.path.join(xml_folder, line))
                line = fp.readline()
        with open(file2, 'r') as fp:
            line = fp.readline()
            while line:
                line = line.rstrip()
                if not line.endswith('.xml'):
                    line = line + '.xml'
                self.file_list.remove(os.path.join(xml_folder, line))
                line = fp.readline()

    def read_bboxes(self):
        for xml_file in self.file_list:
            assert os.path.exists(xml_file)
            self.reader.read_file(xml_file)

        self.boxes = self.reader.get_bboxes()

    def write_classes_file(self, path):
        with open(path, 'w+') as fp:
            for cur_cls in self.classes.keys():
                fp.write(str(self.classes[cur_cls]) + '\t' + str(cur_cls) + '\n')

    def write_oi_csv(self, path):
        self.read_bboxes()
        print(self.boxes[0])
        images = [box['filename'] for box in self.boxes]
        xmins = [box['xmin'] for box in self.boxes]
        xmaxs = [box['xmax'] for box in self.boxes]
        ymins = [box['ymin'] for box in self.boxes]
        ymaxs = [box['ymax'] for box in self.boxes]
        bbox_classes = [box['name'] for box in self.boxes]
        box_class_id = []
        for box_class in bbox_classes:
            try:
                cur_class = self.classes[box_class]
            except KeyError:
                self.classes[box_class] = len(self.classes.keys())
                cur_class = self.classes[box_class]
            box_class_id.append(cur_class)

        annotations = OrderedDict([('ImageID', images), ('Source', ['freeform'] * len(images)),
                                   ('LabelName', box_class_id), ('Confidence', [1.0] * len(images)),
                                   ('XMin', [xmin for xmin in xmins]), ('XMax', [xmax for xmax in xmaxs]),
                                   ('YMin', [ymin for ymin in ymins]), ('YMax', [ymax for ymax in ymaxs])])

        write_annotations_to_openimages_csv(annotations, path)




if __name__ == '__main__':
    """
    Executes the main training loop
    """
    parser = argparse.ArgumentParser(description='General VOC to Openimages converter')
    parser.add_argument('--root', help='root dir', required=True, type=str)
    args = parser.parse_args()

    root_dir = args.root
    train_list = os.path.join(root_dir, 'ImageSets/Main/', 'train.txt')
    test_list = os.path.join(root_dir, 'ImageSets/Main/', 'test.txt')
    xml_folder = os.path.join(root_dir, 'Annotations/')
    converter = OpenImagesWriterFromVOC()
    converter.read_xmls(train_list, xml_folder)
    if not os.path.exists(os.path.join(root_dir, 'OI_Annotations')):
        os.makedirs(os.path.join(root_dir, 'OI_Annotations'))
    print(os.path.join(root_dir, 'OI_Annotations'))
    converter.write_oi_csv(os.path.join(root_dir, 'OI_Annotations/train.csv'))
    converter.reset()
    converter.read_xmls(test_list, xml_folder)
    converter.write_oi_csv(os.path.join(root_dir, 'OI_Annotations/test.csv'))
    converter.write_classes_file(os.path.join(root_dir, 'OI_Annotations/class_map.csv'))


