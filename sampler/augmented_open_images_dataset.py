# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import math
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .datum import Datum
from .dataset import Dataset
from .image import PathImage
from .annotation import ClassificationAnnotation
from .annotation import DetectionAnnotation
from .annotation import BoundingBox, is_valid_coordinates
from .class_map import CSVClassMapIO
from .generate_anchors import generate_anchors
from .iou import iou

import logging
logger = logging.getLogger(__name__)


def create_datum_helper(image_path, bbox_data):
    """Helper to create Datum

    Assumes that if any bounding box coordinates specified is negative
    in the Open Images format (usually -1), then the entry is a
    classification label rather than a detection label.

    Parameters
    ----------
    image_path : str
        full path to image
    bbox_data : list
        list of tuples where each tuple is ((xmin, ymin, xmax, ymax, class_id), (gt_xmin, gt_ymin, gt_xmax, gt_ymax), iou)
        should have at least one element

    Returns
    -------
    Datum
    """
    image = PathImage(image_path)
    if any(coord < 0 for coord in bbox_data[0][0][:4]):
        annotation = ClassificationAnnotation(bbox_data[0][0][4])
    else:
        annotation = DetectionAnnotation([BoundingBox(*bbox_tuple, metadata={'gt_bbox': gt_bbox_tuple, 'iou': iou}) for bbox_tuple, gt_bbox_tuple, iou in bbox_data])
    return Datum(image, [annotation])


def create_data_job_helper(image_ids_list, image_paths,
        annotations, job_index, num_loading_workers):
    """Helper to create data (image ID -> Datum)

    Parameters
    ----------
    image_ids_list : list
        list of image IDs
    image_paths : dict
        image_id -> full path to image
    annotations : dict
        image_id -> list of tuples where each tuple is (xmin, ymin, xmax, ymax, class_id)
    job_index : int
        job index, should be 0 <= job_index < num_loading_workers
    num_loading_workers : int
        number of parallel loading workers

    Returns
    -------
    dict : image ID -> Datum
    """
    chunk_len = int(math.ceil(len(image_ids_list) / num_loading_workers))
    subset_image_ids_list = image_ids_list[chunk_len*job_index:chunk_len*(job_index+1)]
    data = {image_id : create_datum_helper(image_paths[image_id], annotations[image_id]) for image_id in subset_image_ids_list}
    return data


def create_data_parallel_job_helper(image_paths, annotations, num_loading_workers):
    """Helper to create data (image ID -> Datum) in parallel

    Parameters
    ----------
    image_paths : dict
        image_id -> full path to image
    annotations : dict
        image_id -> list of tuples where each tuple is (xmin, ymin, xmax, ymax, class_id)
    num_loading_workers : int
        number of parallel loading workers

    Returns
    -------
    dict : image ID -> Datum
    """
    image_ids_list = list(annotations.keys())
    results = Parallel(n_jobs=num_loading_workers)(delayed(create_data_job_helper)(image_ids_list, image_paths, annotations, job_index, num_loading_workers) for job_index in range(num_loading_workers))
    data = {}
    for d in results:
        data.update(d)
    return data


def annotation_file_contains_label_names(annotation_file):
    """Checks if the annotation file contains label names or class IDs.

    Parameters
    ----------
    annotation_file : str
        path to annotation file

    Returns
    -------
    True if file contains label names (not class IDs)
    """
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f, delimiter=AugmentedOpenImagesDataset.DELIMITER)
        for idx, row in enumerate(reader):
            # get header names and skip
            if idx == 0:
                column_keys = row
                continue

            # get data from each column
            image_entry_data = {}
            for col_idx, entry in enumerate(row):
                image_entry_data[column_keys[col_idx]] = entry

            # test value
            label_name_data = image_entry_data[AugmentedOpenImagesDataset.LABEL_NAME]
            try:
                int(float(label_name_data))
            except ValueError:
                return True

    return False


class AugmentedOpenImagesDataset(Dataset):
    """AugmentedOpenImagesDataset loads a dataset in Open Images format with extra proposals and IOU data

    This format stores proposals for each bounding box and their IOU scores.

    Loading this dataset requires 1) an annotation file specified in the
    Open Images format and 2) a csv file that maps class names to class IDs.
    The annotations file follows the Open Images format. A few notes:
    1) The image IDs are image file names relative to some root
    (which can be specified in initialization), and 2) the labels are
    class IDs (if class names, specify read_as_label_names=True).
    The csv file contains two columns: the first column is the class ID
    and the second column is the class name.

    Parameters
    ----------
    annotation_file : str
        path to annotation csv file
    class_map_file : str
        path to class map csv file
    images_root : str
        path to root for images
    metadata : dict (default {})
        arbitrary dataset metadata, just used for book keeping if needed
    read_as_label_names : bool or None (default None)
        if True, force reading LabelName column as a label name
        if False, force reading LabelName column as a class ID
        if None, automatically determine whether label name or class ID
    skip_invalid_bboxes : bool (default True)
        if True, skip invalid bboxes, otherwise throw error
    num_loading_workers : int (default 8)
        number of workers for loading data from csv for potential speedups
    """
    DELIMITER = ','
    IMAGE_ID = 'ImageID'
    IMAGE_WIDTH = 'ImageWidth'
    IMAGE_HEIGHT = 'ImageHeight'
    LABEL_NAME = 'LabelName'
    GTXMIN = 'GTXMin'
    GTYMIN = 'GTYMin'
    GTXMAX = 'GTXMax'
    GTYMAX = 'GTYMax'
    XMIN = 'XMin'
    YMIN = 'YMin'
    XMAX = 'XMax'
    YMAX = 'YMax'
    IOU = 'IOU'

    def __init__(self, annotation_file, class_map_file, images_root=None,
            metadata={}, read_as_label_names=None, skip_invalid_bboxes=True,
            num_loading_workers=16):
        assert isinstance(annotation_file, str), type(annotation_file)
        assert os.path.exists(annotation_file), annotation_file
        assert isinstance(class_map_file, str), type(class_map_file)
        assert os.path.exists(class_map_file), class_map_file
        if images_root is not None:
            assert isinstance(images_root, str), type(images_root)
            assert os.path.exists(images_root), images_root
        assert isinstance(metadata, dict), type(metadata)
        assert isinstance(num_loading_workers, int), type(num_loading_workers)
        assert num_loading_workers >= 1, num_loading_workers

        if read_as_label_names is None:
            read_as_label_names = annotation_file_contains_label_names(annotation_file)

        # read in class map file
        class_map = CSVClassMapIO.read(class_map_file)

        # parse annotations data
        annotations = defaultdict(list) # image_id -> [(xmin, ymin, xmax, ymax, class_id)]
        image_paths = {} # image_id -> path
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f, delimiter=AugmentedOpenImagesDataset.DELIMITER)
            for idx, row in enumerate(reader):
                # get header names and skip
                if idx == 0:
                    column_keys = row
                    continue

                # get data from each column
                image_entry_data = {}
                for col_idx, entry in enumerate(row):
                    image_entry_data[column_keys[col_idx]] = entry

                image_id = image_entry_data[AugmentedOpenImagesDataset.IMAGE_ID]
                class_name = image_entry_data[AugmentedOpenImagesDataset.LABEL_NAME]
                xmin = float(image_entry_data[AugmentedOpenImagesDataset.XMIN])
                ymin = float(image_entry_data[AugmentedOpenImagesDataset.YMIN])
                xmax = float(image_entry_data[AugmentedOpenImagesDataset.XMAX])
                ymax = float(image_entry_data[AugmentedOpenImagesDataset.YMAX])
                gt_xmin = float(image_entry_data[AugmentedOpenImagesDataset.GTXMIN])
                gt_ymin = float(image_entry_data[AugmentedOpenImagesDataset.GTYMIN])
                gt_xmax = float(image_entry_data[AugmentedOpenImagesDataset.GTXMAX])
                gt_ymax = float(image_entry_data[AugmentedOpenImagesDataset.GTYMAX])
                iou = float(image_entry_data[AugmentedOpenImagesDataset.IOU])

                if not is_valid_coordinates(xmin, ymin, xmax, ymax):
                    message = 'invalid coordinates: {}: {}'.format(image_id, (xmin, ymin, xmax, ymax))
                    if skip_invalid_bboxes:
                        logger.warn('skipping ' + message)
                        continue
                    else:
                        raise ValueError(message)

                if not is_valid_coordinates(gt_xmin, gt_ymin, gt_xmax, gt_ymax):
                    message = 'invalid groundtruth coordinates: {}: {}'.format(image_id, (gt_xmin, gt_ymin, gt_xmax, gt_ymax))
                    if skip_invalid_bboxes:
                        logger.warn('skipping ' + message)
                        continue
                    else:
                        raise ValueError(message)

                class_id = class_map.get_index(class_name) if read_as_label_names else int(float(class_name))
                annotations[image_id].append(((xmin, ymin, xmax, ymax, class_id), (gt_xmin, gt_ymin, gt_xmax, gt_ymax), iou))
                image_paths[image_id] = os.path.join(images_root, image_id) if images_root is not None else image_id

        # put data into desired format
        data = create_data_parallel_job_helper(image_paths, annotations, num_loading_workers)
        super(AugmentedOpenImagesDataset, self).__init__(data, metadata=metadata)


class AugmentedOpenImagesDatasetWriter(object):
    """AugmentedOpenImagesDatasetWriter converts OpenImagesDataset into AugmentedOpenImagesDataset format

    Parameters
    ----------
    dataset : OpenImagesDataset
    """
    def __init__(self, dataset):
        self._dataset = dataset

    def write(self, out_file):
        """Write to output file.

        Parameters
        ----------
        out_file : str
            path to output file
        """
        with open(out_file, 'w') as f:
            header_format_string = '{delim}'.join(['{imageid}',
                '{imagewidth}', '{imageheight}', '{labelname}',
                '{gtxmin}', '{gtymin}', '{gtxmax}', '{gtymax}',
                '{xmin}', '{ymin}', '{xmax}', '{ymax}', '{iou}']) + '\n'
            f.write(header_format_string.format(
                delim=AugmentedOpenImagesDataset.DELIMITER,
                imageid=AugmentedOpenImagesDataset.IMAGE_ID,
                imagewidth=AugmentedOpenImagesDataset.IMAGE_WIDTH,
                imageheight=AugmentedOpenImagesDataset.IMAGE_HEIGHT,
                labelname=AugmentedOpenImagesDataset.LABEL_NAME,
                gtxmin=AugmentedOpenImagesDataset.GTXMIN,
                gtymin=AugmentedOpenImagesDataset.GTYMIN,
                gtxmax=AugmentedOpenImagesDataset.GTXMAX,
                gtymax=AugmentedOpenImagesDataset.GTYMAX,
                xmin=AugmentedOpenImagesDataset.XMIN,
                ymin=AugmentedOpenImagesDataset.YMIN,
                xmax=AugmentedOpenImagesDataset.XMAX,
                ymax=AugmentedOpenImagesDataset.YMAX,
                iou=AugmentedOpenImagesDataset.IOU,
            ))
            with tqdm(total=len(self._dataset)) as progress_bar:
                for image_id in self._dataset.image_ids:
                    datum = self._dataset[image_id]
                    for bbox in datum.annotations[0].data:
                        im_height, im_width, _ = datum.image.data.shape
                        gt_bbox = (bbox.class_id,) + bbox.get_relative_coordinates(im_width, im_height)

                        proposals, ious = self._generate_proposals(datum.image.data, bbox.get_absolute_coordinates(im_width, im_height))
                        for p, i in zip(proposals, ious):
                            p_rel = (1.*p[0]/im_width, 1.*p[1]/im_height, 1.*p[2]/im_width, 1.*p[3]/im_height)
                            f.write('{},{},{},{},{},{}\n'.format(image_id,
                                                                 im_width,
                                                                 im_height,
                                                                 ','.join(['{0:.3f}'.format(x) for x in gt_bbox]),
                                                                 ','.join(['{0:.3f}'.format(x) for x in p_rel]),
                                                                 '{0:.2f}'.format(i)))
                    progress_bar.update(1)

    def _get_bbox(self, bbox):
        xcenter, ycenter, bbox_width, bbox_height = bbox
        xmin = xcenter - 0.5 * (bbox_width - 1)
        ymin = ycenter - 0.5 * (bbox_height - 1)
        xmax = xcenter + 0.5 * (bbox_width - 1)
        ymax = ycenter + 0.5 * (bbox_height - 1)
        return (xmin, ymin, xmax, ymax)

    def _is_valid_proposal(self, bbox, img):
        image_height, image_width, _ = img.shape
        xmin, ymin, xmax, ymax = bbox

        return xmin >= 0 and ymin >= 0 and xmax < image_width and ymax < image_height

    def _generate_proposals(self, img, bbox, ratios=[0.5, 1, 2], scales=[0.5, 1, 2, 4, 7, 10, 13, 16]):
        image_height, image_width, _ = img.shape
        xmin, ymin, xmax, ymax = bbox
        bbox_width = xmax-xmin+1
        bbox_height = ymax-ymin+1
        xcenter = xmin + 0.5*(bbox_width-1)
        ycenter = ymin + 0.5*(bbox_height-1)

        # get anchors
        min_dim = min(bbox_width, bbox_height)
        anchors = generate_anchors(base_size=min_dim, ratios=ratios, scales=scales)

        # get proposals
        proposals = np.array([self._get_bbox((xcenter, ycenter, a[2]-a[0]+1, a[3]-a[1]+1)) for a in anchors])
        proposals = [list(int(pp) for pp in p) for p in proposals if self._is_valid_proposal(p, img)]

        ious = []
        for p in proposals:
            iou_val = iou(bbox, tuple(p))
            ious.append(iou_val)

        return proposals, ious


class DetOpenImagesDatasetWriter(object):
    """AugmentedOpenImagesDatasetWriter converts OpenImagesDataset into AugmentedOpenImagesDataset format

    Parameters
    ----------
    dataset : OpenImagesDataset
    """
    def __init__(self, dataset, target_sizes, max_sizes, feat_stride=16):
        self._dataset = dataset

        self.target_sizes = []
        self.max_sizes = []
        for target_size, max_size in zip(target_sizes, max_sizes):
            self.target_sizes.append(target_size)
            self.max_sizes.append(max_size)

        self.feat_stride = feat_stride

    @staticmethod
    def clip_boxes(boxes, im_shape):
        """
        Clip boxes to image boundaries.
        :param boxes: [N, 4* num_classes]
        :param im_shape: tuple of 2
        :return: [N, 4* num_classes]
        """
        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    @staticmethod
    def filter_boxes(boxes, min_size):
        """
        filter small boxes.
        :param boxes: [N, 4 * num_classes]
        :param min_size:
        :return: keep:
        """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    def write(self, out_file):
        """Write to output file.

        Parameters
        ----------
        out_file : str
            path to output file
        """
        with open(out_file, 'w') as f:
            header_format_string = '{delim}'.join(['{imageid}',
                '{imagewidth}', '{imageheight}', '{labelname}',
                '{gtxmin}', '{gtymin}', '{gtxmax}', '{gtymax}',
                '{xmin}', '{ymin}', '{xmax}', '{ymax}', '{iou}']) + '\n'
            f.write(header_format_string.format(
                delim=AugmentedOpenImagesDataset.DELIMITER,
                imageid=AugmentedOpenImagesDataset.IMAGE_ID,
                imagewidth=AugmentedOpenImagesDataset.IMAGE_WIDTH,
                imageheight=AugmentedOpenImagesDataset.IMAGE_HEIGHT,
                labelname=AugmentedOpenImagesDataset.LABEL_NAME,
                gtxmin=AugmentedOpenImagesDataset.GTXMIN,
                gtymin=AugmentedOpenImagesDataset.GTYMIN,
                gtxmax=AugmentedOpenImagesDataset.GTXMAX,
                gtymax=AugmentedOpenImagesDataset.GTYMAX,
                xmin=AugmentedOpenImagesDataset.XMIN,
                ymin=AugmentedOpenImagesDataset.YMIN,
                xmax=AugmentedOpenImagesDataset.XMAX,
                ymax=AugmentedOpenImagesDataset.YMAX,
                iou=AugmentedOpenImagesDataset.IOU,
            ))
            with tqdm(total=len(self._dataset)) as progress_bar:
                for image_id in self._dataset.image_ids:
                    gt_boxes = datum.annotations[0].data
                    datum = self._dataset[image_id]
                    im_shape = datum.image.data.shape[0:2]

                    im_height, im_width = im_shape

                    im_size_min = np.min(im_shape)
                    im_size_max = np.max(im_shape)
                    for target_size, max_size in zip(self.target_sizes, self.max_sizes):
                        im_scale = float(target_size) / float(im_size_min)
                        # prevent bigger axis from being more than max_size:
                        if np.round(im_scale * im_size_max) > max_size:
                            im_scale = float(max_size) / float(im_size_max)

                        proposals, ious = self._generate_proposals(im_shape * im_scale, gt_boxes)
                        for p, i in zip(proposals, ious):
                            p_rel = (1.*p[0]/im_width, 1.*p[1]/im_height, 1.*p[2]/im_width, 1.*p[3]/im_height)
                            f.write('{},{},{},{},{},{}\n'.format(image_id,
                                                                 im_width,
                                                                 im_height,
                                                                 ','.join(['{0:.3f}'.format(x) for x in gt_bbox]),
                                                                 ','.join(['{0:.3f}'.format(x) for x in p_rel]),
                                                                 '{0:.2f}'.format(i)))
                    progress_bar.update(1)

    def _get_bbox(self, bbox):
        xcenter, ycenter, bbox_width, bbox_height = bbox
        xmin = xcenter - 0.5 * (bbox_width - 1)
        ymin = ycenter - 0.5 * (bbox_height - 1)
        xmax = xcenter + 0.5 * (bbox_width - 1)
        ymax = ycenter + 0.5 * (bbox_height - 1)
        return (xmin, ymin, xmax, ymax)

    def _is_valid_proposal(self, bbox, img):
        image_height, image_width, _ = img.shape
        xmin, ymin, xmax, ymax = bbox

        return xmin >= 0 and ymin >= 0 and xmax < image_width and ymax < image_height

    def _generate_proposals(self, im_shape, gt_boxes, ratios=[0.5, 1, 2], scales=[0.5, 1, 2, 4, 7, 10, 13, 16]):

        base_anchors = generate_anchors(base_size=self.feat_stride, ratios=ratios,
                                        scales=scales)
        num_anchors = base_anchors.shape[0]
        im_height, im_width = im_shape
        shift_x = np.arange(0, im_width) * self.feat_stride
        shift_y = np.arange(0, im_height) * self.feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, copy=False)

        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()

        K = shifts.shape[0]
        all_anchors = base_anchors.reshape((1, num_anchors, 4)) + \
                      shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * num_anchors, 4))
