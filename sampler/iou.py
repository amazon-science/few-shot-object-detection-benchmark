# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import logging
logger = logging.getLogger(__name__)


def iou(bbox1, bbox2):
    """Compute intersection over union score between two bounding boxes.

    Parameters
    ----------
    bbox1 : tuple
        (xmin, ymin, xmax, ymax)
    bbox2 : tuple
        (xmin, ymin, xmax, ymax)

    Returns
    -------
    float : IOU score
    """
    assert isinstance(bbox1, tuple), type(bbox1)
    assert len(bbox1) == 4, len(bbox1)
    assert isinstance(bbox2, tuple), type(bbox2)
    assert len(bbox2) == 4, len(bbox2)

    x1min, y1min, x1max, y1max = bbox1
    x2min, y2min, x2max, y2max = bbox2
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    intersection = max((xmax-xmin+1), 0) * max((ymax-ymin+1), 0)
    bbox1_area = (x1max-x1min+1)*(y1max-y1min+1)
    bbox2_area = (x2max-x2min+1)*(y2max-y2min+1)

    iou = intersection/(bbox1_area + bbox2_area - intersection)
    return iou
