# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .image import Image
from .annotation import Annotation

import logging
logger = logging.getLogger(__name__)


class Datum(object):
    """Datum is a container for image and annotations data

    Datum supports a list of Annotion objects for multiple tasks.

    Parameters
    ----------
    image :
        Image object
    annotations_list : list
        list of Annotation objects
    """
    def __init__(self, image, annotations_list):
        assert issubclass(image.__class__, Image), type(image)
        assert isinstance(annotations_list, list), type(annotations_list)
        assert all(issubclass(a.__class__, Annotation) for a in annotations_list)

        self._image = image # Image object
        self._annotations_list = annotations_list # list of Annotation objects

    @property
    def image(self):
        """Get the Image object

        Returns
        -------
        Image
        """
        return self._image

    @property
    def annotations(self):
        """Get the list of Annotation objects

        Returns
        -------
        list of Annotation objects
        """
        return self._annotations_list
