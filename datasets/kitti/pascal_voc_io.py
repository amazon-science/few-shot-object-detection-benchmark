# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs

XML_EXT = '.xml'
ENCODE_METHOD = 'utf-8'


class PascalVocReader:

    def __init__(self):
        self.bboxes = []
        self.verified = False

    def read_file(self, file):
        self.file = file
        self.parse_XML()

    def reset(self):
        self.bboxes = []
        self.verified = False


    def get_bboxes(self):
        return self.bboxes

    def add_bbxes(self, filename, label, bndbox, difficult, truncated, pose, occluded, groupof, depiction, inside):
        cur_bbox = {}
        cur_bbox['xmin'] = float(bndbox.find('xmin').text)
        cur_bbox['xmax'] = float(bndbox.find('xmax').text)
        cur_bbox['ymin'] = float(bndbox.find('ymin').text)
        cur_bbox['ymax'] = float(bndbox.find('ymax').text)

        cur_bbox['occluded'] = occluded
        cur_bbox['truncated'] = truncated
        cur_bbox['groupof'] = groupof
        cur_bbox['depiction'] = depiction
        cur_bbox['inside'] = inside
        cur_bbox['difficult'] = difficult
        cur_bbox['pose'] = pose
        cur_bbox['name'] = label
        cur_bbox['filename'] = filename
        if cur_bbox['ymax'] > cur_bbox['ymin'] and cur_bbox['xmax'] > cur_bbox['xmin']:
            self.bboxes.append(cur_bbox)

    def parse_XML(self):
        assert self.file.endswith(XML_EXT), "Unsupport file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xmltree = ElementTree.parse(self.file, parser=parser).getroot()

        try:
            path = xmltree.find('path').text
        except AttributeError:
            path = xmltree.find('filename').text
            if '.' not in path:
                path = path + '.jpg'
        try:
            verified = xmltree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xmltree.findall('object'):
            bndbox = object_iter.find("bndbox")
            label = object_iter.find('name').text
            # Add chris
            difficult = False
            truncated = False
            pose = False
            occluded = False
            groupof = False
            depiction = False
            inside = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            if object_iter.find('truncated') is not None:
                try:
                    truncated = bool(int(object_iter.find('truncated').text))
                except ValueError:
                    truncated = False
            if object_iter.find('pose') is not None:
                if object_iter.find('pose').text != 'Unspecified':
                    pose = object_iter.find('pose').text
            if object_iter.find('occluded') is not None:
                occluded = bool(int(object_iter.find('occluded').text))
            if object_iter.find('groupof') is not None:
                groupof = bool(int(object_iter.find('groupof').text))
            if object_iter.find('depiction') is not None:
                depiction = bool(int(object_iter.find('depiction').text))
            if object_iter.find('inside') is not None:
                inside = bool(int(object_iter.find('inside').text))

            self.add_bbxes(path, label, bndbox, difficult, truncated, pose,
                           occluded, groupof, depiction, inside)