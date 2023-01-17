# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .datum import Datum

import logging
logger = logging.getLogger(__name__)


class Dataset(object):
    """Dataset is a generic dataset containing images and their annotations

    Dataset provides a general interface for loading common.
    For loading a specific dataset, create a new class
    that is a subclass of Dataset.

    A subclass of Dataset just needs to provide Dataset with a dictionary
    of image IDs to Datum where a Datum holds image and annotations data.

    Parameters
    ----------
    data : dict
        image ID -> Datum
    metadata : dict (default {})
        arbitrary dataset metadata, just used for book keeping if needed
    """
    def __init__(self, data, metadata={}):
        assert isinstance(data, dict), type(data)
        assert all(issubclass(d.__class__, Datum) for d in data.values())
        assert isinstance(metadata, dict), type(metadata)

        self._data = data # image ID -> Datum
        self.metadata = metadata # user can set after initialized

    def __len__(self):
        """The length of the dataset is the number of images

        Returns
        -------
        int : number of images
        """
        return len(self._data)

    def __getitem__(self, image_id):
        """Get Datum object for a particular image_id

        Parameters
        ----------
        image_id :
            image ID to retrieve Datum object

        Returns
        -------
        Datum
        """
        return self._data[image_id]

    @property
    def image_ids(self):
        """Return the set of image IDs

        Returns
        -------
        dict_keys
        """
        return self._data.keys()

    @property
    def num_images(self):
        """Get the number of images in the dataset

        This is a convenience function for __len__().

        Returns
        -------
        int : number of images
        """
        return len(self)
