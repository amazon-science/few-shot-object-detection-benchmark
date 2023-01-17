# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict, Counter

import numpy as np
import logging
import random

logger = logging.getLogger(__name__)


def consolidate_detection_samples(dataset, indices, classes):
    """Helper to consolidate the results of n-way k-shot sampling on a detection dataset

    Parameters
    ----------
    dataset : DetectionImagePathsDataset
    indices : list
        list of image indices (int) for dataset, generated from WayShotSampler.sample()
    classes : set
        set of n-way classes (int), generated from WayShotSampler.sample()
        any bounding boxes in images that are not part of the n-way classes are dropped

    Returns
    -------
    list with each element (image_id, (xmin, ymin, xmax, ymax, label))
    """
    data = []
    for i in indices:
        img_path, labels_list, bbox_coords_list = dataset[i]
        for label, bbox in zip(labels_list, bbox_coords_list):
            if label in classes:
                data.append((img_path, bbox + (label,)))
    return sorted(data)


class WayShotSampler(object):
    DISCARD = 'discard'
    ROLLOVER = 'rollover'
    LAST_OPTIONS = [DISCARD, ROLLOVER]
    MODES = ['cls', 'nat1', 'nat2', 'ratio', 'balanced_instance']

    # 'cls': standard N-way K-shot
    # 'nat1': natural N-way K-shot ; K <- num_shots
    # 'nat2': natural N-way K-shot ; K <- num_shots / num_ways
    # 'ratio': K <- num_per_class * num_shots
    # 'balanced_instance': balanced sampling similar to VOC/COCO

    def __init__(self, dataset, num_ways=None, num_shots=None,
                 diversity=False, last=ROLLOVER, shuffle=True, mode='nat1',
                 ignore_shots_deficiency=False):
        if num_ways is not None:
            assert isinstance(num_ways, int), type(num_ways)
            assert num_ways > 0, num_ways
        if num_shots is not None:
            assert isinstance(num_shots, int), type(num_shots)
            assert num_shots > 0, num_shots
        assert isinstance(last, str), type(last)
        assert last in WayShotSampler.LAST_OPTIONS, last
        assert mode in WayShotSampler.MODES, mode

        self._dataset = dataset
        self._num_ways = num_ways
        self._num_shots = num_shots
        self._diversity = diversity
        self._last = last
        self._shuffle = shuffle
        self._mode = mode

        # build class to samples map
        self._class_to_samples_map = defaultdict(set)
        # build class to samples counter, which maps class name to a counter
        # where counter = image id to frequency of occurence of that class in that image id
        self._class_to_samples_counter = defaultdict(list)
        # build samples to class counter, which maps image id to a counter
        # where counter = class to frequency of occurence of that class in that image id
        self._samples_to_class_counter = defaultdict(Counter)

        for idx, data in enumerate(self._dataset):
            target = data[1]
            if isinstance(target, list):
                for t in target:
                    self._class_to_samples_map[t].add(idx)
                    self._class_to_samples_counter[t].append(idx)
                    self._samples_to_class_counter[idx][t] += 1
            else:
                self._class_to_samples_map[target].add(idx)
                self._class_to_samples_counter[t].append(idx)
                self._samples_to_class_counter[idx][t] += 1
        self._class_to_samples_map = {k: list(v) for k, v in self._class_to_samples_map.items()}

        # build class to samples counter sorted w.r.t. frequency
        for k in self._class_to_samples_counter:
            v = self._class_to_samples_counter[k]
            v = Counter(v)
            v = [(imgid, freq) for imgid, freq in v.items()]
            random.shuffle(v)
            self._class_to_samples_counter[k] = v

        if self._mode == 'ratio':
            ratio = num_shots
            self._num_shots = defaultdict(int)
            for k in self._class_to_samples_map:
                self._num_shots[k] = max(1, round(len(self._class_to_samples_map[k]) * ratio))

        assert len(self._class_to_samples_map) > 0, 'could not find any classes with sufficient number of samples'
        assert self._num_ways is None or self._num_ways <= len(
            self._class_to_samples_map), 'num of ways must be <= num of classes'

        # classes for current __iter__
        self._current_classes = set()
        # remaining classes for next calls to __iter__; only useful for diversity
        self._remaining_classes = set(self._class_to_samples_map.keys())

    def sample(self, classes=False, seed=None):
        """Sample N classes and K shots, automatically or manually

        Parameters
        ----------
        classes : set or None or False
            * set of classes manually specified to sample, or
            * None means manually sample all classes, or
            * False means do not manually sample (default behavior).
            this mode is only available when not in diversity mode

        Returns
        -------
        indices, classes
            where indices is a list of sampled indices in dataset
            and classes is a set of classes sampled
        """
        if seed is None:
            self._initialize_indices(classes)
        else:
            state = np.random.get_state()
            np.random.seed(seed)
            self._initialize_indices(classes)
            np.random.set_state(state)
        return self._current_indices, self._current_classes

    def _initialize_indices(self, manually_sampled_classes=False):
        """Helper to initialize sample indices

        Parameters
        ----------
        manually_sampled_classes : set or None or False
            * set of classes manually specified to sample, or
            * None means manually sample all classes, or
            * False means do not manually sample (default behavior).
            this mode is only available when not in diversity mode
        """
        self._current_indices = set()

        # perform N-way/class sampling
        if manually_sampled_classes is not False:
            assert not self._diversity, 'diversity option cannot be set if manually specifying classes'
            if manually_sampled_classes is None:
                self._current_classes = self._remaining_classes
            else:
                assert isinstance(manually_sampled_classes, set), type(manually_sampled_classes)
                assert (self._num_ways is None) or (len(
                    manually_sampled_classes) == self._num_ways), 'num ways must match length of manually-specified classes'
                for c in manually_sampled_classes:
                    if c not in self._class_to_samples_map.keys():
                        print(c)
                assert all(c in self._class_to_samples_map.keys() for c in
                           manually_sampled_classes), 'classes specified must exist in the dataset'

                self._current_classes = manually_sampled_classes
        elif self._num_ways is None:
            self._current_classes = self._remaining_classes
        elif not self._diversity:
            self._current_classes = set(np.random.choice(list(self._remaining_classes), self._num_ways, replace=False))
        elif len(self._remaining_classes) < self._num_ways:
            if self._last == WayShotSampler.DISCARD:
                # reset classes and sample
                self._remaining_classes = set(self._class_to_samples_map.keys())
                self._current_classes = set(
                    np.random.choice(list(self._remaining_classes), self._num_ways, replace=False))
                self._remaining_classes = self._remaining_classes.difference(self._current_classes)
            elif self._last == WayShotSampler.ROLLOVER:
                # choose the rest of the remaining classes
                self._current_classes = self._remaining_classes
                # reset the remaining classes
                self._remaining_classes = set(self._class_to_samples_map.keys()).difference(self._current_classes)
                # choose from the rest of the resetted remaining classes
                num_to_rollover = self._num_ways - len(self._current_classes)
                rollover_classes = set(np.random.choice(list(self._remaining_classes), num_to_rollover, replace=False))
                self._current_classes.update(rollover_classes)
                self._remaining_classes = self._remaining_classes.difference(rollover_classes)
        else:
            self._current_classes = set(np.random.choice(list(self._remaining_classes), self._num_ways, replace=False))
            self._remaining_classes = self._remaining_classes.difference(self._current_classes)

        if (self._num_shots is not None) and ('nat' in self._mode):
            # sample the indices of samples in the dataset given N-way and class-agnostic K-shot
            samples_all = set()
            for c in self._current_classes:
                samples_all.update(self._class_to_samples_map[c])

            if self._mode == 'nat1':
                dlen = self._num_shots * len(self._class_to_samples_map)
            else:
                dlen = self._num_shots

            # len(dataset) <= N*K: take all samples
            if len(samples_all) <= dlen:
                samples = samples_all
                dlen = len(samples_all)

            # len(dataset) > N*K
            else:
                samples = np.random.choice(list(samples_all), dlen, replace=False)
                # samples = np.random.choice(list(range(len(self._dataset))), self._num_shots, replace=False)
                samples = set(samples)

                # compute sampled class distribution
                sampled_class_to_samples_map = defaultdict(set)
                for idx in samples:
                    target = self._dataset[idx][1]
                    if isinstance(target, list):
                        for t in target:
                            sampled_class_to_samples_map[t].add(idx)
                    else:
                        sampled_class_to_samples_map[target].add(idx)
                sampled_class_to_samples_map = {k: list(v) for k, v in sampled_class_to_samples_map.items()}

                # guarantee at least 1-shot for each class
                new_samples = set()
                keep_samples = set()
                for c in self._current_classes:
                    if sampled_class_to_samples_map.get(c) is None:
                        idx = np.random.choice(self._class_to_samples_map[c], 1, replace=False)
                        new_samples.add(idx.item())
                    else:
                        idx = np.random.choice(sampled_class_to_samples_map[c], 1, replace=False)
                        keep_samples.add(idx.item())

                # remove samples from oversampled classes
                if len(new_samples) > 0:
                    del_samples = np.random.choice(list(samples - keep_samples), len(new_samples), replace=False)
                    samples.difference_update(del_samples)
                    samples.update(new_samples)


            self._current_indices.update(samples)
        # balanced instance level k-shot sampling
        # sample k' images for class c such that k instances of class c are present in k' images
        elif self._num_shots is not None and self._mode == 'balanced_instance':
            cls_count = Counter()
            for c in self._current_classes:
                samples = []
                for sample in self._class_to_samples_counter[c]:
                    imgid, cfreq = sample
                    samples.append(imgid)
                    for c_ in self._samples_to_class_counter[imgid]:
                        cls_count[c_] += self._samples_to_class_counter[imgid][c_]
                    if cls_count[c] >= self._num_shots:
                        break
                for sample in samples:
                    self._current_indices.add(sample)
        else:
            # sample the indices of samples in the dataset given N-way and K-shot
            for c in self._current_classes:
                samples = self._class_to_samples_map[c]
                if self._num_shots is not None:
                    if (self._mode == 'cls') and (self._num_shots < len(samples)):
                        samples = np.random.choice(samples, self._num_shots, replace=False)
                    elif (self._mode == 'ratio') and (self._num_shots[c] < len(samples)):
                        samples = np.random.choice(samples, self._num_shots[c], replace=False)
                for sample in samples:
                    self._current_indices.add(sample)
        self._current_indices = list(self._current_indices)
        # shuffle or keep in order of classes
        if self._shuffle:
            np.random.shuffle(self._current_indices)

    def __iter__(self):
        self._initialize_indices()
        return iter(self._current_indices)

    def __len__(self):
        return len(self._current_indices)