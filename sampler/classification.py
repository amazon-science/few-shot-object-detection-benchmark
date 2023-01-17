# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import mxnet as mx
from PIL import Image

from .dataset import Dataset

import logging
logger = logging.getLogger(__name__)


class ClassificationDataset(mx.gluon.data.Dataset):
    """ClassificationDataset is a wrapper around Dataset to provide classification labels for data loaders

    Samples provided from this class follow a classification format where
    each sample is annotated with a single class ID.

    Samples with multiple bounding boxes from Dataset are split into multiple
    cropped samples.

    Since a dataset can support multiple annotations per sample,
    this class simplifies by only considering one annotation per sample
    so that a sample only needs to consider one list of bounding boxes
    instead of multiple lists of bounding boxes. The splitting into
    multiple samples is simpler in this case.

    Example Usage:

    ```
    train_dataset = ClassificationDataset(
        OpenImagesDataset(annotation_file, class_map_file)
        )

    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform_first(transform_train),
        batch_size=batch_size, shuffle=True,
        last_batch='discard', num_workers=num_workers)
    ```

    Parameters
    ----------
    dataset : Dataset
    annotation_index : int (default 0)
        samples can have multiple annotations (as a list),
        so this selects the index of the annotations to consider
    transform : callable or None (default None)
        function that takes an image and label (a class ID),
        and produces a transformed image and label
    """
    def __init__(self, dataset, annotation_index=0, transform=None):
        assert issubclass(dataset.__class__, Dataset), type(dataset)
        assert isinstance(annotation_index, int), type(annotation_index)
        assert annotation_index >= 0, annotation_index

        self._dataset = dataset
        self._transform = transform

        self._annotation_index = annotation_index
        # list of (image_id, index) tuples where index maps
        # bounding box index for the sample, or None if no bounding box
        self._samples = []
        for image_id in dataset.image_ids:
            annotations = dataset[image_id].annotations
            if annotation_index < len(annotations):
                ann = annotations[annotation_index].data
                if isinstance(ann, list):
                    self._samples.extend([(image_id, idx) for idx, _ in enumerate(ann)])
                else:
                    self._samples.append((image_id, None))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        """Get sample given an index

        Parameters
        ----------
        index : int

        Returns
        -------
        img, target where
            img is an MXNet NDArray, and
            target is an integer (class ID)
        """
        image_id, bbox_index = self._samples[index]
        datum = self._dataset[image_id]
        ann = datum.annotations[self._annotation_index]
        bbox = ann.data[bbox_index] if bbox_index is not None else None
        target = ann.data[bbox_index].class_id if bbox_index is not None else ann.data

        img = Image.fromarray(datum.image.data)
        if bbox is not None:
            width, height = img.size
            img = img.crop(bbox.get_absolute_coordinates(width, height))
        img = mx.nd.array(img)

        if self._transform is not None:
            img, target = self._transform(img, target)

        return img, target

    @property
    def dataset(self):
        return self._dataset


class ClassificationImagePathsDataset(ClassificationDataset):
    """ClassificationImagePathsDataset is like ClassificationDataset but returns (image path, target, bbox) instead of (raw image, target)

    Parameters
    ----------
    dataset : Dataset
    annotation_index : int (default 0)
        samples can have multiple annotations (as a list),
        so this selects the index of the annotations to consider
    transform : callable or None (default None)
        function that takes an image (image path str!), a label (a class ID)
        and bbox coordinates (xmin, ymin, xmin, ymax),
        and produces a transformed image and label
    """
    def __init__(self, dataset, annotation_index=0, transform=None):
        super(ClassificationImagePathsDataset, self).__init__(dataset=dataset,
                                                              annotation_index=annotation_index,
                                                              transform=transform)

    def __getitem__(self, index):
        """Get sample given an index

        Parameters
        ----------
        index : int

        Returns
        -------
        img, target, bbox where
            img is the a string (path to the image),
            target is an integer (class ID), and
            bbox is a tuple (xmin, ymin, xmax, ymax)
        """
        image_id, bbox_index = self._samples[index]
        datum = self._dataset[image_id]
        img = datum.image.path
        ann = datum.annotations[self._annotation_index]
        bbox = ann.data[bbox_index].coordinates if bbox_index is not None else None
        target = ann.data[bbox_index].class_id if bbox_index is not None else ann.data

        if self._transform is not None:
            img, target, bbox = self._transform(img, target, bbox)

        return img, target, bbox
