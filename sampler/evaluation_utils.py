# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
from collections import OrderedDict

annotations_headers = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax']
extras_headers = ['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
all_headers = annotations_headers + extras_headers


def write_annotations_to_openimages_csv(sampled_data, cur_map, path='annotations.csv'):
    all_images = []
    all_bbox_classes = []
    all_boxes = []
    for data in sampled_data:
        img_path = data[0]
        label = data[1][4]
        bbox_coords = data[1][0:4]
        all_boxes.append(bbox_coords)
        all_images.append(img_path)
        all_bbox_classes.append(cur_map[label])

    annotations = OrderedDict([('ImageID', all_images), ('Source', ['freeform'] * len(all_boxes)),
                               ('LabelName', all_bbox_classes), ('Confidence', [1.0] * len(all_boxes)),
                               ('XMin', [bbox[0] for bbox in all_boxes]), ('XMax', [bbox[2] for bbox in all_boxes]),
                               ('YMin', [bbox[1] for bbox in all_boxes]),
                               ('YMax', [bbox[3] for bbox in all_boxes])])

    for header in annotations_headers:
        assert header in annotations
    df = pd.DataFrame(annotations)
    anno_len = len(annotations['ImageID'])
    extras = OrderedDict()
    for header in extras_headers:
        extras[header] = [-1] * anno_len
    df_extra = pd.DataFrame(extras)
    df = pd.concat([df, df_extra], axis=1)
    df.to_csv(path, index=False, float_format='%.3f')


def get_class_names(path):
    logos = {}
    with open(path) as fp:
        line = fp.readline()
        while line:
            processed_line = line.rstrip().split('\t')
            logos[int(processed_line[0])] = processed_line[1]
            line = fp.readline()

    return logos


def get_class_id(path):
    logos = {}
    with open(path) as fp:
        line = fp.readline()
        while line:
            processed_line = line.rstrip().split('\t')
            logos[str(processed_line[1])] = int(processed_line[0])
            line = fp.readline()

    return logos


def write_class_map(class_map, cur_map, path):
    logos = {}
    with open(path, 'w+') as fp:
        for logo in cur_map.keys():
            fp.write(str(cur_map[logo])+'\t'+str(class_map[logo])+'\n')
            logos[cur_map[logo]] = class_map[logo]

    return logos


def overlaps(gt_bbox, proposals):
    """
    :param gt_bbox:  np array, in order of x1, y1, x2, y2
    :param proposals: np array, in order of x1, x2, y1, y2
    :return: 
    """
    #assert gt_bbox[0] < gt_bbox[2]
    #assert gt_bbox[1] < gt_bbox[3]

    overlaps = []

    for proposal in proposals:
        if proposal[0] >= proposal[2] or proposal[1] >= proposal[3]:
            iou = 0.0
            overlaps.append(iou)
            continue
        #assert proposal[0] < proposal[2]
        #assert proposal[1] < proposal[3]

        x_left = max(gt_bbox[0], proposal[0])
        y_top = max(gt_bbox[1], proposal[1])
        x_right = min(gt_bbox[2], proposal[2])
        y_bottom = min(gt_bbox[3], proposal[3])

        if x_right < x_left or y_bottom < y_top:
            iou = 0.0
            overlaps.append(iou)
            continue

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bbox_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
        proposal_area = (proposal[2] - proposal[0]) * (proposal[3] - proposal[1])

        iou = intersection_area / float(bbox_area + proposal_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        overlaps.append(iou)

    return np.array(overlaps)


def compute_average_recall(unsorted_overlaps):
    """
    :param unsorted_overlaps: np array of unsorted overlaps
    :return: average recall
    """
    all_overlaps = np.sort(unsorted_overlaps)
    num_pos = np.size(all_overlaps)
    dx = 0.001

    overlap = np.arange(0, 1, dx)
    overlap[-1] = 1
    recall = np.zeros((len(overlap), 1))
    for i in range(len(overlap)):
        recall[i] = float(np.sum(all_overlaps >= overlap[i])) / num_pos

    good_recall = recall[overlap >= 0.5].transpose()
    avg_recall = 2 * dx * np.trapz(good_recall)
    return avg_recall, float(np.sum(all_overlaps >= 0.5)) / num_pos


def greedy_matching_rowwise(iou_matrix):
    assert (iou_matrix.shape[0] <= iou_matrix.shape[1])
    n = iou_matrix.shape[0]
    matching = np.zeros((n, 1))
    objective = 0
    for ii in range(n):
        max_per_row = np.max(iou_matrix, axis=1)
        max_col_per_row = np.argmax(iou_matrix, axis=1)
        max_iou = np.max(max_per_row)
        row = np.argmax(max_per_row)
        if max_iou == -np.inf:
            break
        objective = objective + max_iou
        col = max_col_per_row[row]
        matching[row] = col
        iou_matrix[row, :] = -np.inf
        iou_matrix[:, col] = -np.inf
    return matching, objective


def greedy_matching(iou_matrix, gt_boxes, proposals):
    n = iou_matrix.shape[0]
    m = iou_matrix.shape[1]
    assert (n == len(gt_boxes))
    assert (m == len(proposals))
    out_matrix = iou_matrix.copy()
    if n > m:
        gt_matching = greedy_matching_rowwise(out_matrix.transpose())
        proposal_matching = np.arange(m).transpose()
    else:
        gt_matching = np.arange(n).transpose()
        proposal_matching, _ = greedy_matching_rowwise(out_matrix)
    
    best_overlap = np.zeros((n, 1))
    best_boxes = np.zeros((n, 4))
    for pair_idx in range(np.size(gt_matching)):
        gt_idx = gt_matching[pair_idx].astype('int')
        proposal_idx = proposal_matching[pair_idx].astype('int')
        try:
            best_overlap[gt_idx] = iou_matrix[gt_idx, proposal_idx]
        except ValueError:
            pass
            # print gt_idx, proposal_idx
        best_boxes[gt_idx, :] = proposals[proposal_idx, :]

    return best_overlap, best_boxes


def closest_proposals(gt_boxes, proposals):
    """
    Find the best overlaps
    :param gt_boxes: gt_boxes of each image
    :param proposals: proposals of each image
    :return: best_over laps and the corresponding best boxes
    """

    num_gt_boxes = gt_boxes.shape[0]
    num_candidates = proposals.shape[0]

    iou_matrix = np.zeros(shape=(num_gt_boxes, num_candidates))
    for ii in range(num_gt_boxes):
        iou = overlaps(gt_boxes[ii], proposals)
        iou_matrix[ii, :] = iou

    best_overlap, best_boxes = greedy_matching(iou_matrix, gt_boxes, proposals)
    return best_overlap, best_boxes


def match_proposals(proposals, gt_boxes, gt_box_classes, iou_thres=0.5):
    """
    Find the best overlaps
    :param gt_boxes: gt_boxes of each image
    :param proposals: proposals of each image
    :return: proposals classes
    """
    num_candidates = proposals.shape[0]

    proposal_classes = np.zeros(shape=(num_candidates, ))
    for ii in range(num_candidates):
        iou = overlaps(proposals[ii], gt_boxes)
        if np.max(iou) < iou_thres:
            proposal_classes[ii] = 0
        else:
            # shift class by 1
            proposal_classes[ii] = gt_box_classes[np.argmax(iou)] + 1

    return proposal_classes



