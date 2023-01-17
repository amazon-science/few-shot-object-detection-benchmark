# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from numbers import Number

import logging
logger = logging.getLogger(__name__)


def is_valid_coordinates(xmin, ymin, xmax, ymax):
    """Returns whether the bounding box coordinates provided are valid

    This function checks whether the coordinates provided are all absolute
    or relative but not both. It also checks whether the min coordinates
    are smaller than the max coordinates.

    Parameters
    ----------
    xmin : int or float
        leftmost coordinate, relative or absolute; should be >= 0
    ymin : int or float
        topmost coordinate, relative or absolute; should be >= 0
    xmax : int or float
        rightmost coordinate, relative or absolute; should be <= 1.0 if relative
    ymax : int or float
        bottommost coordinate, relative or absolute; should be <= 1.0 if relative

    Returns
    -------
    bool : True if coordinates are all relative or all absolute but not both
    """
    min_max_constraints = xmin < xmax and ymin < ymax
    return min_max_constraints


class BoundingBox(object):
    """BoundingBox stores the coordinates and class ID of a single bounding box

    BoundingBox has support for relative and absolute coordinates
    (but not both simultaneously).

    Parameters
    ----------
    xmin : int or float
        leftmost coordinate, relative or absolute; should be >= 0
    ymin : int or float
        topmost coordinate, relative or absolute; should be >= 0
    xmax : int or float
        rightmost coordinate, relative or absolute; should be <= 1.0 if relative
    ymax : int or float
        bottommost coordinate, relative or absolute; should be <= 1.0 if relative
    class_id : int
        class ID
    metadata : dict (default {})
        additional metadata about the bounding box (e.g., difficulty)
    """
    def __init__(self, xmin, ymin, xmax, ymax, class_id, metadata={}):
        assert isinstance(xmin, Number), xmin
        assert xmin >= 0, xmin
        assert isinstance(ymin, Number), ymin
        assert ymin >= 0, ymin
        assert isinstance(xmax, Number), xmax
        assert isinstance(ymax, Number), ymax
        assert isinstance(class_id, int), class_id
        assert isinstance(metadata, dict), metadata
        assert is_valid_coordinates(xmin, ymin, xmax, ymax), (xmin, ymin, xmax, ymax)

        self._xmin = float(xmin)
        self._ymin = float(ymin)
        self._xmax = float(xmax)
        self._ymax = float(ymax)
        self._class_id = class_id
        self._metadata = metadata

    def get_absolute_coordinates(self, width, height):
        """Get absolute coordinates given the width and height of the image

        This function is only useful if the bounding box contains
        relative coordinates. If the bounding box already contains
        absolute coordinates, then it just returns that.

        Parameters
        ----------
        width : int
        height : int

        Returns
        -------
        (xmin, ymin, xmax, ymax) each as int
        """
        if not self.has_relative_coordinates:
            return self.coordinates
        else:
            return (int(self.xmin*width), int(self.ymin*height), int(self.xmax*width), int(self.ymax*height))

    def get_relative_coordinates(self, width, height):
        """Get relative coordinates given the width and height of the image

        This function is only useful if the bounding box contains
        absolute coordinates. If the bounding box already contains
        relative coordinates, then it just returns that.

        Parameters
        ----------
        width : int
        height : int

        Returns
        -------
        (xmin, ymin, xmax, ymax) each as float
        """
        if self.has_relative_coordinates:
            return self.coordinates
        else:
            return (1.*self.xmin/width, 1.*self.ymin/height, 1.*self.xmax/width, 1.*self.ymax/height)

    @property
    def has_relative_coordinates(self):
        """Whether the bounding box contains relative coordinates or not

        Relative coordinates is defined as having all coordinate values
        between 0 and 1.

        Returns
        -------
        True if bounding box contains relative coordinates
        """
        return self._xmin >= 0 and self._xmin <= 1 \
               and self._ymin >= 0 and self._ymin <= 1 \
               and self._xmax >= 0 and self._xmax <= 1 \
               and self._ymax >= 0 and self._ymax <= 1

    @property
    def xmin(self):
        """Get leftmost coordinate

        Returns
        -------
        int if absolute coordinates otherwise float
        """
        return int(self._xmin) if not self.has_relative_coordinates else self._xmin

    @property
    def ymin(self):
        """Get topmost coordinate

        Returns
        -------
        int if absolute coordinates otherwise float
        """
        return int(self._ymin) if not self.has_relative_coordinates else self._ymin

    @property
    def xmax(self):
        """Get rightmost coordinate

        Returns
        -------
        int if absolute coordinates otherwise float
        """
        return int(self._xmax) if not self.has_relative_coordinates else self._xmax

    @property
    def ymax(self):
        """Get bottommost coordinate

        Returns
        -------
        int if absolute coordinates otherwise float
        """
        return int(self._ymax) if not self.has_relative_coordinates else self._ymax

    @property
    def coordinates(self):
        """Get coordinates

        Returns ints if absolute coordinates otherwise floats

        Returns
        -------
        (xmin, ymin, xmax, ymax)
        """
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def class_id(self):
        """Get class ID

        Returns
        -------
        int
        """
        return self._class_id

    @property
    def coordinates_and_class_id(self):
        """Get coordinates and class ID in one tuple

        Returns
        -------
        (xmin, ymin, xmax, ymax, class ID)
        """
        return self.coordinates + (self.class_id,)

    @property
    def metadata(self):
        """Get additional metadata about the bounding box

        Returns
        -------
        dict
        """
        return self._metadata

    def __repr__(self):
        return 'BoundingBox({},{},{},{},{})'.format(*self.coordinates_and_class_id)


class Annotation(object):
    """Annotation is associated with a sample
    """
    @property
    def data(self):
        """Get the underlying annotation data
        """
        raise NotImplementedError()


class ClassificationAnnotation(Annotation):
    """ClassificationAnnotation is a class ID associated with a sample

    Parameters
    ----------
    class_id : int
        class_index
    """
    def __init__(self, class_id):
        assert isinstance(class_id, int), type(class_id)
        self._data = class_id

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return 'ClassificationAnnotation({})'.format(self._data)


class DetectionAnnotation(Annotation):
    """DetectionAnnotation is a list of bounding boxes associated with a sample

    Parameters
    ----------
    bbox_list : list
        list of bounding boxes
    """
    def __init__(self, bbox_list):
        assert isinstance(bbox_list, list), type(bbox_list)
        assert all(isinstance(bbox, BoundingBox) for bbox in bbox_list)
        self._data = bbox_list

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return 'DetectionAnnotation({})'.format(self._data)
