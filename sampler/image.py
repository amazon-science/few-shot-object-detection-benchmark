# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from PIL import Image as PILImage

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)


def assert_valid_numpy_image_data(data):
    """Assert whether the provided numpy array is in a valid image format

    This function will only succeed if the provided numpy array is a
    valid raw RGB image before any preprocessing/normalization.

    Parameters
    ----------
    data : np.ndarray
    """
    assert isinstance(data, np.ndarray), type(data)
    assert data.ndim == 3, data.shape
    assert data.shape[2] == 3, data.shape
    assert data.dtype == 'uint8', data.dtype
    assert (data >= 0).all() and (data <= 255).all(), 'all elements must be between 0 and 255'


class Image(object):
    """Image is a container for image data
    """
    @property
    def data(self):
        """Get the underlying image data as a numpy array

        The image data must be a valid raw RGB image before any
        preprocessing/normalization.

        Returns
        -------
        np.ndarray :
            * shape : H x W x 3
            * dtype : uint8
            * values : 0 to 255
        """
        raise NotImplementedError()


class RawImage(Image):
    """RawImage is a container for an image in numpy form

    Parameters
    ----------
    data : np.ndarray
        numpy array containing raw image data
    """
    def __init__(self, data):
        assert_valid_numpy_image_data(data)
        self._data = data

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return 'RawImage(shape={})'.format(self._data.shape)


class PathImage(Image):
    """PathImage is a container for an image on disk

    The image is loaded when requested through the data property.

    Parameters
    ----------
    path : str
        path to image file
    ignore_check : bool (default False)
        if True, ignore path existence check, which could save time
        or be good for debugging
    """
    def __init__(self, path, ignore_check=True):
        assert isinstance(path, str), type(path)
        assert isinstance(ignore_check, bool), type(ignore_check)
        assert ignore_check or os.path.exists(path), path
        self._path = path

    @property
    def data(self):
        with PILImage.open(self._path) as img:
            img_rgb = img.convert('RGB')
            data = np.asarray(img_rgb)
        assert_valid_numpy_image_data(data)
        return data

    @property
    def path(self):
        return self._path

    def __repr__(self):
        return "PathImage('{}')".format(self._path)
