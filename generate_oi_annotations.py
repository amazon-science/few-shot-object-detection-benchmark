# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import copy
import itertools
import numpy as np

from sampler.open_images_dataset import OpenImagesDataset
from sampler.detection import DetectionImagePathsDataset
from sampler.way_shot import WayShotSampler, consolidate_detection_samples
from sampler.evaluation_utils import get_class_names, write_class_map, write_annotations_to_openimages_csv

datasets = [
'logodet_3k',
'clipart',
'oktoberfest',
'fashionpedia',
'kitti',
'visdrone2019',
'deepfruits',
'iwildcam',
'crowdhuman',
'sixray',
]

output_path = './few_shot_sampling'
seeds = range(10)
shots = [1, 3, 5, 10]
test_shots = [1000]
max_num_classes = 50
train_seed_offset = 0
test_seed_offset = 1000
class_seed_offset = 2000
log_file = 'debug_gen_oi_anno_all_100.txt'
log_file2 = 'debug_gen_oi_anno_class_100.txt'

class OpenImagesDatasetClassRemover(object):

    def __init__(self, dataset: OpenImagesDataset, annotation_index=0):
        self._dataset = dataset
        self._annotation_index = annotation_index
        self._samples = list(dataset.image_ids)
        self.image_to_class = dict()
        self.class_to_image = dict()

        for image_id in self._dataset.image_ids:
            for bbox_anno in self._dataset._data[image_id].annotations[self._annotation_index].data:

                if self.image_to_class.get(image_id) is None:
                    self.image_to_class[image_id] = {bbox_anno.class_id}
                else:
                    self.image_to_class[image_id].add(bbox_anno.class_id)

                if self.class_to_image.get(bbox_anno.class_id) is None:
                    self.class_to_image[bbox_anno.class_id] = {image_id}
                else:
                    self.class_to_image[bbox_anno.class_id].add(image_id)

    @property
    def classes(self):
        return sorted(self.class_to_image)

    @property
    def num_classes(self):
        return len(self.class_to_image)

    def remove(self, c):
        # return if c is already removed
        if c not in self.class_to_image:
            return

        # remove images in c from dataset
        for image_id in sorted(self.class_to_image[c]):

            # remove image_id from class_to_image
            for cc in sorted(self.image_to_class[image_id]):
                self.class_to_image[cc].remove(image_id)

            # remove image_id from dataset
            self._dataset._data.pop(image_id)

        # remove empty class from class_to_image
        for cc in sorted(self.class_to_image):
            if len(self.class_to_image[cc]) == 0:
                self.class_to_image.pop(cc)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=False)

for dataset in datasets:
    if not os.path.exists(os.path.join(output_path, dataset)):
        os.makedirs(os.path.join(output_path, dataset), exist_ok=False)
    dataset_folder = './datasets/'

    dataset_dir = os.path.join(dataset_folder, dataset)
    train_image_list = '{}/OI_Annotations/train.csv'.format(dataset_dir)
    test_image_list = '{}/OI_Annotations/test.csv'.format(dataset_dir)
    class_map = '{}/OI_Annotations/class_map.csv'.format(dataset_dir)
    image_folder = '{}/JPEGImages/'.format(dataset_dir)
    way = None

    if not os.path.isdir(dataset_dir):
        print(f'missing {dataset}')
        continue

    train_oi_dataset = OpenImagesDataset(train_image_list, class_map, images_root=image_folder)
    test_oi_dataset = OpenImagesDataset(test_image_list, class_map, images_root=image_folder)

    train_dataset = DetectionImagePathsDataset(train_oi_dataset)
    targets = [datum[1] for datum in train_dataset]
    train_classes_all = np.unique(np.array(list(itertools.chain.from_iterable(targets)))).tolist()

    test_dataset = DetectionImagePathsDataset(test_oi_dataset)
    targets = [datum[1] for datum in test_dataset]
    test_classes_all = np.unique(np.array(list(itertools.chain.from_iterable(targets)))).tolist()

    # all classes
    all_class_map = get_class_names(class_map)

    # remove classes from train_dataset if they are not in test_dataset
    missing_classes = sorted(set(all_class_map) - set(train_classes_all).intersection(test_classes_all))
    if len(missing_classes) > 0:
        write_me = f'{dataset} has no train classes {sorted(set(all_class_map) - set(train_classes_all))} test classes {sorted(set(all_class_map) - set(test_classes_all))}'
        print(write_me)
        with open(log_file2, 'a') as f:
            f.write(write_me + '\n')

        train_class_remover = OpenImagesDatasetClassRemover(train_oi_dataset)
        test_class_remover = OpenImagesDatasetClassRemover(test_oi_dataset)

        for c in missing_classes:
            train_class_remover.remove(c)
            test_class_remover.remove(c)

            # make sure that train and test classes are the same
            while not (train_class_remover.classes == test_class_remover.classes):
                if train_class_remover.num_classes > test_class_remover.num_classes:
                    for cc in sorted(set(train_class_remover.classes) - set(test_class_remover.classes)):
                        train_class_remover.remove(cc)
                else:
                    for cc in sorted(set(test_class_remover.classes) - set(train_class_remover.classes)):
                        test_class_remover.remove(cc)

        train_oi_dataset = train_class_remover._dataset
        test_oi_dataset = test_class_remover._dataset
        all_classes_safe = train_class_remover.classes
        num_classes_before_sample = train_class_remover.num_classes
    else:
        all_classes_safe = sorted(all_class_map)
        num_classes_before_sample = len(all_class_map)

    for jj in seeds:
        num_classes = num_classes_before_sample

        if num_classes > max_num_classes:

            state = np.random.get_state()
            np.random.seed(class_seed_offset + jj)
            remove_order = np.random.permutation(sorted(test_classes_all)).tolist()
            np.random.set_state(state)

            train_class_remover = OpenImagesDatasetClassRemover(copy.deepcopy(train_oi_dataset))
            test_class_remover = OpenImagesDatasetClassRemover(copy.deepcopy(test_oi_dataset))

            for c in remove_order:
                train_class_remover.remove(c)
                test_class_remover.remove(c)

                # make sure that train and test classes are the same
                while not (train_class_remover.classes == test_class_remover.classes):
                    if train_class_remover.num_classes > test_class_remover.num_classes:
                        for cc in sorted(set(train_class_remover.classes) - set(test_class_remover.classes)):
                            train_class_remover.remove(cc)
                    else:
                        for cc in sorted(set(test_class_remover.classes) - set(train_class_remover.classes)):
                            test_class_remover.remove(cc)

                if train_class_remover.num_classes <= max_num_classes:
                    num_classes = train_class_remover.num_classes
                    break

            train_oi_dataset_jj = train_class_remover._dataset
            test_oi_dataset_jj = test_class_remover._dataset
            classes = set(train_class_remover.classes)

        else:
            train_oi_dataset_jj = train_oi_dataset
            test_oi_dataset_jj = test_oi_dataset
            classes = False

        # dataset
        train_dataset = DetectionImagePathsDataset(train_oi_dataset_jj)
        test_dataset = DetectionImagePathsDataset(test_oi_dataset_jj)
        num_train = len(train_dataset)
        num_test = len(test_dataset)

        targets = [datum[1] for datum in train_dataset]
        this_train_classes = np.unique(np.array(list(itertools.chain.from_iterable(targets))))
        # assert num_classes == len(this_train_classes), f'num_classes expected: {num_classes}; train_classes: {len(this_train_classes)}'
        if num_classes != len(this_train_classes):
            write_me = f'num_classes expected: {num_classes}; train_classes: {len(this_train_classes)}'
            with open(log_file2, 'a') as f:
                f.write(write_me + '\n')

        targets = [datum[1] for datum in test_dataset]
        this_test_classes = np.unique(np.array(list(itertools.chain.from_iterable(targets))))
        # assert num_classes == len(this_test_classes), f'num_classes expected: {num_classes}; test_classes: {len(this_test_classes)}'
        if num_classes != len(this_test_classes):
            write_me = f'num_classes expected: {num_classes}; test_classes: {len(this_test_classes)}'
            with open(log_file2, 'a') as f:
                f.write(write_me + '\n')

        this_classes = classes if classes else all_classes_safe
        class_map_path = os.path.join(output_path, dataset, f'class_map_{jj}.csv')
        cur_mapping = dict(zip(this_classes, range(len(this_classes))))
        write_class_map(all_class_map, cur_mapping, class_map_path)

        # train
        for shot in shots:
            postfix = f'nat1_{shot}s'
            print(f'{dataset} train {len(train_dataset)} {shot}s {jj}t')

            train_sampler = WayShotSampler(train_dataset, num_ways=way, num_shots=shot, mode='nat1',
                                           diversity=False, last='rollover', ignore_shots_deficiency=True)
            train_indices, train_classes = train_sampler.sample(seed=train_seed_offset+jj, classes=classes)

            # remove this if always identical
            class_map_path = os.path.join(output_path, dataset, f'class_map_{postfix}_{jj}.csv')
            cur_mapping = dict(zip(train_classes, range(len(train_classes))))
            write_class_map(all_class_map, cur_mapping, class_map_path)

            # keep track of OI Annotations for reproducibility
            current_train_image_list = os.path.join(output_path, dataset, f'train_{postfix}_{jj}.csv')

            # if not os.path.isfile(current_train_image_list):
            sampled_train_data = consolidate_detection_samples(train_dataset, train_indices, train_classes)
            write_annotations_to_openimages_csv(sampled_train_data, cur_mapping, path=current_train_image_list)

            expected = num_classes * shot
            check1 = (num_train <= expected) and (len(train_indices) == num_train)
            check2 = (num_train > expected) and (len(train_indices) == expected)
            write_me = f'{dataset}\ttrain\t{jj}\t{num_train}\t{len(train_indices)}\t{len(train_classes)}\t{check1}\t{check2}\t{(check1 or check2)}'
            with open(log_file, 'a') as f:
                f.write(write_me + '\n')

        # test
        for shot in test_shots:
            postfix = f'nat2_{shot}s'
            print(f'{dataset} test {len(test_dataset)} {shot}s {jj}t')

            test_sampler = WayShotSampler(test_dataset, num_ways=way, num_shots=shot, mode='nat2',
                                          ignore_shots_deficiency=True)
            test_indices, test_classes = test_sampler.sample(seed=test_seed_offset+jj, classes=classes)

            # remove this if always identical
            class_map_path = os.path.join(output_path, dataset, f'class_map_{postfix}_{jj}.csv')
            cur_mapping = dict(zip(test_classes, range(len(test_classes))))
            write_class_map(all_class_map, cur_mapping, class_map_path)

            # keep track of OI Annotations for reproducibility
            current_test_image_list = os.path.join(output_path, dataset, f'test_{postfix}_{jj}.csv')

            # if not os.path.isfile(current_test_image_list):
            sampled_test_data = consolidate_detection_samples(test_dataset, test_indices, test_classes)
            write_annotations_to_openimages_csv(sampled_test_data, cur_mapping, path=current_test_image_list)

            expected = shot
            check1 = (num_test <= expected) and (len(test_indices) == num_test)
            check2 = (num_test > expected) and (len(test_indices) == expected)
            write_me = f'{dataset}\ttest\t{jj}\t{num_test}\t{len(test_indices)}\t{len(test_classes)}\t{check1}\t{check2}\t{(check1 or check2)}'
            with open(log_file, 'a') as f:
                f.write(write_me + '\n')

print('done')