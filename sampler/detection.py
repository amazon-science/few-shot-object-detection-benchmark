# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from PIL import Image

from .dataset import Dataset

import logging
logger = logging.getLogger(__name__)


class DetectionImagePathsDataset(object):
    """DetectionImagePathsDataset is like DetectionDataset but returns (image path, targets, bboxes) instead of (raw image, bboxes/targets)

    Parameters
    ----------
    dataset : Dataset
    annotation_index : int (default 0)
        samples can have multiple annotations (as a list),
        so this selects the index of the annotations to consider
        function that takes an image (image path str!), a list of labels (a class ID)
        and a list of bbox coordinates (xmin, ymin, xmin, ymax),
        and produces a transformed image and label
    """
    def __init__(self, dataset, annotation_index=0, transform=None):
        super(DetectionImagePathsDataset, self).__init__()
        assert issubclass(dataset.__class__, Dataset), type(dataset)
        assert isinstance(annotation_index, int), type(annotation_index)
        assert annotation_index >= 0, annotation_index

        self._dataset = dataset
        self._transform = transform
        self._annotation_index = annotation_index
        self._samples = list(dataset.image_ids)

    def __getitem__(self, index):
        """Get sample given an index

        Parameters
        ----------
        index : int

        Returns
        -------
        img, target, bbox where
            img is the a string (path to the image),
            target is a list of integers (class ID), and
            bbox is a list of tuples (xmin, ymin, xmax, ymax)
        """
        image_id = self._samples[index]
        datum = self._dataset[image_id]
        img = datum.image.path
        ann = datum.annotations[self._annotation_index].data

        if isinstance(ann, list):
            target = [bb.class_id for bb in ann]
            bbox = [bb.coordinates for bb in ann]
        else:
            width, height = Image.fromarray(datum.image.data).size
            target = [ann]
            bbox = [(0, 0, width, height)]

        if self._transform is not None:
            img, target, bbox = self._transform(img, target, bbox)

        return img, target, bbox

    def __len__(self):
        return len(self._dataset)

    @property
    def dataset(self):
        return self._dataset

