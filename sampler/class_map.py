# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os

import logging
logger = logging.getLogger(__name__)


class ClassMap(object):
    """ClassMap stores an association from class ID to class name and vice versa
    """
    def __init__(self):
        self._index_to_name = {} # class ID -> class name
        self._name_to_index = {} # class name -> class ID
        self._max_class_id = -1

    def add(self, class_name, class_id=None):
        assert isinstance(class_name, str), type(class_name)
        assert isinstance(class_id, int) or class_id is None, type(class_id)
        assert not self.is_name_added(class_name)

        if class_id is None:
            if len(self._index_to_name) > 0 and self._max_class_id == -1:
                raise ValueError('class map already initialized with class ID: must specify class_id')
            self._max_class_id += 1
            class_id = self._max_class_id
        elif self._max_class_id != -1:
            raise ValueError('class map already initialized without class ID: cannot use class_id')

        assert not self.is_index_added(class_id)
        self._index_to_name[class_id] = class_name
        self._name_to_index[class_name] = class_id

    def __len__(self):
        """Returns the number of classes

        Returns
        -------
        int : number of classes
        """
        return len(self._index_to_name)

    def is_index_added(self, class_id):
        """Check whether class ID is already added

        Parameters
        ----------
        class_id : int

        Returns
        -------
        bool : True if class ID is already added
        """
        return class_id in self._index_to_name

    def is_name_added(self, class_name):
        """Check whether class name is already added

        Parameters
        ----------
        class_name : str

        Returns
        -------
        bool : True if class name is already added
        """
        return class_name in self._name_to_index

    def get_name(self, class_id):
        """Get class name given the class ID

        Parameters
        ----------
        class_id : int
            class ID

        Returns
        -------
        str : class name
        """
        return self._index_to_name[class_id]

    def get_index(self, class_name):
        """Get class ID given the class name

        Parameters
        ----------
        class_name : str
            class name

        Returns
        -------
        int : class ID
        """
        return self._name_to_index[class_name]

    @property
    def class_ids(self):
        """Get the set of possible class indices

        Returns
        -------
        dict_keys
        """
        return self._index_to_name.keys()


class CSVClassMapIO(object):
    """CSVClassMapIO provides methods for reading and writing class map csv files
    """
    DEFAULT_DELIMITER = '\t'
    HEADER_CLASS_ID = 'class_id'
    HEADER_CLASS_NAME = 'class_name'

    @staticmethod
    def read(csv_file, header=False, delimiter=DEFAULT_DELIMITER):
        """Read class map from csv file

        Parameters
        ----------
        csv_file : str
            path to csv file
        header : bool (default False)
            specify True if csv file has header (and subsequently ignore it)
        delimiter : str (default '\t')
            delimiter for csv file
        Returns
        -------
        ClassMap
        """
        assert isinstance(csv_file, str), type(csv_file)
        assert os.path.exists(csv_file), csv_file
        assert isinstance(delimiter, str), type(delimiter)

        class_map = ClassMap()
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for idx, row in enumerate(reader):
                if header and idx == 0:
                    continue
                class_id, class_name = row
                class_id = int(class_id)
                class_map.add(class_name, class_id)
        return class_map

    @staticmethod
    def write(csv_file, class_map, header=False, delimiter=DEFAULT_DELIMITER):
        """Write class map to csv file

        Parameters
        ----------
        csv_file : str
            path to csv file
        class_map : ClassMap
        header : bool (default False)
            if True, write header
        delimiter : str (default '\t')
            delimiter for csv file
        """
        assert isinstance(csv_file, str), type(csv_file)
        assert issubclass(class_map.__class__, ClassMap)
        assert isinstance(delimiter, str), type(delimiter)

        with open(csv_file, 'w') as f:
            if header:
                f.write('{class_id}{delim}{class_name}\n'.format(class_id=CSVClassMapIO.HEADER_CLASS_ID,
                                                                 delim=delimiter,
                                                                 class_name=CSVClassMapIO.HEADER_CLASS_NAME))
            for class_id in class_map.class_ids:
                f.write('{class_id}{delim}{class_name}\n'.format(class_id=class_id,
                                                                 delim=delimiter,
                                                                 class_name=class_map.get_name(class_id)))
