# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict

import numpy as np
from mxnet import gluon


class BoundingBoxSampler(gluon.data.sampler.Sampler):
    """BoundingBoxSampler is a sampler for easy/hard positive/negative bounding boxes

    Dataset provided should include a get_map() method to get
    the bounding boxes and IOUs

    Should be used with gluon.data.BatchSampler

    Parameters
    ----------
    dataset : Dataset
    shuffle : bool (default True)
        whether to randomly shuffle
    num_samples_per_gt : int (default 1)
        number of samples per ground truth
        (e.g., 2 will double number of samples)
    positives_percent : float (default 0.5)
        percentage of positives vs. negatives
    hard_negatives_percent : float (default 0.5)
        percentage of hard negatives within negatives
    hard_negatives_threshold_range : tuple (default (0.3, 0.4))
        IOU threshold range for hard negatives
    """
    def __init__(self, dataset, shuffle=True, num_samples_per_gt=1,
            positives_percent=0.5, hard_negatives_percent=0.5,
            hard_negatives_threshold_range=(0.3, 0.4), positives_threshold=0.5):
        assert isinstance(num_samples_per_gt, int), type(num_samples_per_gt)
        assert num_samples_per_gt >= 1, num_samples_per_gt
        assert isinstance(positives_percent, float), type(positives_percent)
        assert positives_percent >= 0 and positives_percent <= 1, positives_percent
        assert isinstance(hard_negatives_percent, float), type(hard_negatives_percent)
        assert hard_negatives_percent >= 0 and hard_negatives_percent <= 1, hard_negatives_percent
        assert isinstance(hard_negatives_threshold_range, tuple), type(hard_negatives_threshold_range)
        assert len(hard_negatives_threshold_range) == 2, hard_negatives_threshold_range
        assert all(isinstance(x, float) and x >= 0 and x <= 1 for x in hard_negatives_threshold_range), hard_negatives_threshold_range
        assert hard_negatives_threshold_range[0] <= hard_negatives_threshold_range[1] and hard_negatives_threshold_range[1] < positives_threshold, hard_negatives_threshold_range

        self._dataset = dataset
        self._shuffle = shuffle
        self._num_samples_per_gt = num_samples_per_gt
        self._positives_percent = positives_percent
        self._hard_negatives_percent = hard_negatives_percent
        self._hard_negatives_threshold_range = hard_negatives_threshold_range
        self._positives_threshold = dataset.positives_threshold

        self._samples_map = defaultdict(lambda: defaultdict(list))
        for (image_id, gt_bbox), samples_list in dataset.samples_map.items():
            for sample_id, iou in samples_list:
                if iou >= positives_threshold:
                    self._samples_map[(image_id, gt_bbox)]['positives'].append(sample_id)
                elif iou >= hard_negatives_threshold_range[0] and iou < hard_negatives_threshold_range[1]:
                    self._samples_map[(image_id, gt_bbox)]['hard_negatives'].append(sample_id)
                elif iou < hard_negatives_threshold_range[0]:
                    self._samples_map[(image_id, gt_bbox)]['negatives'].append(sample_id)

        self._initialize_indices()

    def _initialize_indices(self):
        self._current_indices = []
        for (image_id, gt_bbox), samples_dict in self._samples_map.items():
            for num_samples in range(self._num_samples_per_gt):
                if np.random.rand() <= self._positives_percent:
                    sample_index = np.random.choice(samples_dict['positives'])
                elif np.random.rand() <= self._hard_negatives_percent:
                    sample_index = np.random.choice(samples_dict['positives']) if not samples_dict['hard_negatives'] else np.random.choice(samples_dict['hard_negatives'])
                else:
                    sample_index = np.random.choice(samples_dict['positives']) if not samples_dict['negatives'] else np.random.choice(samples_dict['negatives'])
                self._current_indices.append(sample_index)

        if self._shuffle:
            np.random.shuffle(self._current_indices)

    def __iter__(self):
        self._initialize_indices()
        return iter(self._current_indices)

    def __len__(self):
        return len(self._current_indices)
