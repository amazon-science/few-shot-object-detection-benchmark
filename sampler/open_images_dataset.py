# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import math
from collections import defaultdict

from joblib import Parallel, delayed

from .datum import Datum
from .dataset import Dataset
from .image import PathImage
from .annotation import ClassificationAnnotation
from .annotation import DetectionAnnotation
from .annotation import BoundingBox, is_valid_coordinates
from .class_map import CSVClassMapIO

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
        list of tuples where each tuple is (xmin, ymin, xmax, ymax, class_id)
        should have at least one element

    Returns
    -------
    Datum
    """
    image = PathImage(image_path)
    if any(coord < 0 for coord in bbox_data[0][:4]):
        annotation = ClassificationAnnotation(bbox_data[0][4])
    else:
        annotation = DetectionAnnotation([BoundingBox(*bbox_tuple) for bbox_tuple in bbox_data])
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
    """Checks if the annotation file contains label names or class ID

    Parameters
    ----------
    annotation_file : str
        path to annotation file

    Returns
    -------
    True if file contains label names (not class IDs)
    """
    with open(annotation_file, 'r') as f:
        reader = csv.reader(f, delimiter=OpenImagesDataset.DELIMITER)
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
            label_name_data = image_entry_data[OpenImagesDataset.LABEL_NAME]
            try:
                int(label_name_data)
            except ValueError:
                return True

    return False


class OpenImagesDataset(Dataset):
    """OpenImagesDataset loads a dataset in Open Images format

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
    LABEL_NAME = 'LabelName'
    XMIN = 'XMin'
    XMAX = 'XMax'
    YMIN = 'YMin'
    YMAX = 'YMax'

    def __init__(self, annotation_file, class_map_file, images_root=None,
            metadata={}, read_as_label_names=None, skip_invalid_bboxes=True,
            num_loading_workers=16, offset=0):
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
        self.offset = offset

        # parse annotations data
        annotations = defaultdict(list) # image_id -> [(xmin, ymin, xmax, ymax, class_id)]
        image_paths = {} # image_id -> path
        with open(annotation_file, 'r') as f:
            reader = csv.reader(f, delimiter=OpenImagesDataset.DELIMITER)
            for idx, row in enumerate(reader):
                # get header names and skip
                if idx == 0:
                    column_keys = row
                    continue

                # get data from each column
                image_entry_data = {}
                for col_idx, entry in enumerate(row):
                    image_entry_data[column_keys[col_idx]] = entry

                image_id = image_entry_data[OpenImagesDataset.IMAGE_ID]
                class_name = image_entry_data[OpenImagesDataset.LABEL_NAME]
                xmin = float(image_entry_data[OpenImagesDataset.XMIN])
                ymin = float(image_entry_data[OpenImagesDataset.YMIN])
                xmax = float(image_entry_data[OpenImagesDataset.XMAX])
                ymax = float(image_entry_data[OpenImagesDataset.YMAX])

                if not ( all(coord >= 0 for coord in [xmin, ymin, xmax, ymax]) and is_valid_coordinates(xmin, ymin, xmax, ymax) ):
                    message = 'invalid coordinates: {}: {}'.format(image_id, (xmin, ymin, xmax, ymax))
                    if skip_invalid_bboxes:
                        logger.warn('skipping ' + message)
                        continue
                    else:
                        raise ValueError(message)

                class_id = class_map.get_index(class_name) if read_as_label_names else int(class_name)
                annotations[image_id].append((xmin, ymin, xmax, ymax, class_id))
                image_paths[image_id] = image_id

        # put data into desired format
        data = create_data_parallel_job_helper(image_paths, annotations, num_loading_workers)
        super(OpenImagesDataset, self).__init__(data, metadata=metadata)
