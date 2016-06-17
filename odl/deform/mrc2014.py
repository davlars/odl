# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Definition of the MRC2014 specification in a machine-readable way.

    See [Che+2015]_ or the `explanations on the CCP4 homepage
    <http://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ for the
    text of the specification.

.. [Che+2015] Cheng, A, Henderson, R, Mastronarde, D, Ludtke, S J,
   Schoenmakers, R H M, Short, J, Marabini, R, Dallakyan, S, Agard, D,
   and Winn, M. *MRC2014: Extensions to the MRC format header for electron
   cryo-microscopy and tomography*. Journal of Structural Biology,
   129 (2015), pp 146--150.
"""

import csv
import numpy as np
import struct


__all__ = ('MRCFileReader',)


MRC_2014_SPEC = """
+---------+-------+---------+--------+-------------------------------+
|Long word|Byte   |Data type|Name    |Description                    |
+=========+=======+=========+========+===============================+
|1        |1-4    |Int32    |NX      |Number of columns              |
+---------+-------+---------+--------+-------------------------------+
|2        |5-8    |Int32    |NY      |Number of rows                 |
+---------+-------+---------+--------+-------------------------------+
|3        |9-12   |Int32    |NZ      |Number of sections             |
+---------+-------+---------+--------+-------------------------------+
|4        |13-16  |Int32    |MODE    |Data type                      |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|8        |29-32  |Int32    |MX      |Number of intervals along X of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|9        |33-36  |Int32    |MY      |Number of intervals along Y of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|10       |37-40  |Int32    |MZ      |Number of intervals along Z of |
|         |       |         |        |the "unit cell"                |
+---------+-------+---------+--------+-------------------------------+
|11-13    |41-52  |Float32  |CELLA   |Cell dimension in angstroms    |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|20       |77-80  |Float32  |DMIN    |Minimum density value          |
+---------+-------+---------+--------+-------------------------------+
|21       |81-84  |Float32  |DMAX    |Maximum density value          |
+---------+-------+---------+--------+-------------------------------+
|22       |85-88  |Float32  |DMEAN   |Mean density value             |
+---------+-------+---------+--------+-------------------------------+
|23       |89-92  |Int32    |ISPG    |Space group number 0, 1, or 401|
+---------+-------+---------+--------+-------------------------------+
|24       |93-96  |Int32    |NSYMBT  |Number of bytes in extended    |
|         |       |         |        |header                         |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|27       |105-108|Char     |EXTTYPE |Extended header type           |
+---------+-------+---------+--------+-------------------------------+
|28       |109-112|Int32    |NVERSION|Format version identification  |
|         |       |         |        |number                         |
+---------+-------+---------+--------+-------------------------------+
|...      |       |         |        |                               |
+---------+-------+---------+--------+-------------------------------+
|50-52    |197-208|Float32  |ORIGIN  |Origin in X, Y, Z used in      |
|         |       |         |        |transform                      |
+---------+-------+---------+--------+-------------------------------+
|53       |209-212|Char     |MAP     |Character string 'MAP' to      |
|         |       |         |        |identify file type             |
+---------+-------+---------+--------+-------------------------------+
|54       |213-216|Char     |MACHST  |Machine stamp                  |
+---------+-------+---------+--------+-------------------------------+
|55       |217-220|Float32  |RMS     |RMS deviation of map from mean |
|         |       |         |        |density                        |
+---------+-------+---------+--------+-------------------------------+
"""
# TODO: add nlabl stuff

MRC_HEADER_BYTES = 1024
WORD_LENGTH = 4
# Add more if needed
TYPE_MAP_MRC2NPY = {
    'Float64': np.dtype('float64'),
    'Float32': np.dtype('float32'),
    'Int64': np.dtype('int64'),
    'Int32': np.dtype('int32'),
    'Char': np.dtype('S1')}
TYPE_MAP_NPY2MRC = {v: k for k, v in TYPE_MAP_MRC2NPY.items()}

# TODO: add rest
DTYPE_MAP_NPY2STRUCT = {
    np.dtype('float32'): 'f',
    np.dtype('int32'): 'i',
    np.dtype('int16'): 'h',
    np.dtype('int8'): 'b',
    np.dtype('uint64'): 'L',
    np.dtype('uint32'): 'I',
    np.dtype('uint16'): 'H',
    np.dtype('uint8'): 'B',
    np.dtype('S1'): 'b'
    }


MODE2DTYPE = {
    0: np.dtype('uint8'),
    1: np.dtype('int16'),
    2: np.dtype('float32'),
    6: np.dtype('uint16')
    }

ANGSTROM = 1e-10


def reformat_spec(spec):
    """Return the reformatted specification table.

    The given specification is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.

    Parameters
    ----------
    spec : `str`
        Specification given as a string containing a definition table

    Returns
    -------
    lines : `list`
        Table with leading and trailing '|' stripped and lines containing
        '...' removed, as one list entry per line
    """
    return [line[1:-1].rstrip() for line in spec.splitlines()
            if line.startswith('|') and '...' not in line]


def mrc_spec_fields(spec):
    """Read the specification and return a list of fields.

    The given specification is assumed to be in
    `reST grid table format
    <http://docutils.sourceforge.net/docs/user/rst/quickref.html#tables>`_.

    Parameters
    ----------
    spec : `str`
        Specification given as a string containing a definition table

    Returns
    -------
    fields : `list` of `dict`
        Field list of the specification with combined multi-line entries
    """
    spec_lines = reformat_spec(spec)

    # Guess the CSV dialect and read the table, producing an iterable
    dialect = csv.Sniffer().sniff(spec_lines[0], delimiters='|')
    reader = csv.DictReader(reformat_spec(MRC_2014_SPEC), dialect=dialect)

    # Read the fields as dictionaries and transform keys and values to
    # lowercase.
    fields = []
    for row in reader:
        new_row = {}
        if row['Long word'].strip():
            # Start of a new field, indicated by a nontrivial 'Long word' entry
            for key, val in row.items():
                new_row[key.strip()] = val.strip()
            fields.append(new_row)
        else:
            # We have the second row of a multi-line field. We
            # append all stripped values to the corresponding existing entry
            # value with an extra space.

            if not fields:
                # Just to make sure that this situation did not happen at
                # the very beginning of the table
                continue

            for key, val in row.items():
                fields[-1][key.strip()] += (' ' + val).rstrip()

    return fields


def mrc_standardized_fields(field_list):
    """Convert the field keys and values to standard format.

    Data type is piped through `numpy.dtype`, and name is converted
    to lowercase. All keys are converted to lowercase.

    The standardized fields are as follows:

    +-------------+---------+------------------------------------------+
    |Name         |Data type|Description                               |
    +=============+=========+==========================================+
    |'name'       |string   |Name of the element                       |
    +-------------+---------+------------------------------------------+
    |'description'|string   |Description of the element                |
    +-------------+---------+------------------------------------------+
    |'byte start' |int      |Offset of the current element in bytes    |
    +-------------+---------+------------------------------------------+
    |'byte size'  |int      |Size of the current element in bytes      |
    +-------------+---------+------------------------------------------+
    |'dtype'      |type     |Data type of the current element as       |
    |             |         |defined by Numpy                          |
    +-------------+---------+------------------------------------------+
    |'dshape'     |tuple    |For multi-elements: number of elements per|
    |             |         |dimension. Can be empty for single        |
    |             |         |elements.                                 |
    +-------------+---------+------------------------------------------+

    Parameters
    ----------
    field_list : sequence of dicts
        Dictionaries describing the field elements.

    Returns
    -------
    conv_list : tuple of sdicts
        List of standardized fields.
    """
    # Parse the fields and represent them in a unfied way
    conv_list = []
    for row, field in enumerate(field_list):
        new_field = {}

        # Name and description: lowercase name, copy description
        new_field['name'] = field['Name'].lower()
        new_field['description'] = field['Description']

        # Get offset from long word range
        num_range = field['Long word'].split('-')
        nstart = int(num_range[0])
        nend = int(num_range[-1])  # 0 for range of type 3-

        byte_offset = (nstart - 1) * WORD_LENGTH
        last_byte = nend * WORD_LENGTH - 1  # -1 for nend == 0

        # Get byte range, check consistency with row id and set start
        byte_range = field['Byte'].split('-')
        byte_start = int(byte_range[0]) - 1
        byte_end = int(byte_range[-1]) - 1
        if byte_offset != byte_start:
            raise ValueError(
                'in row {}: byte offset {} from long word id not equal '
                'to start byte {} from the byte range.'
                ''.format(row + 1, byte_offset, byte_start))
        if last_byte != -1 and last_byte != byte_end:
            raise ValueError(
                'in row {}: position {} of last byte in long words {} not '
                'equal to end byte {} from the byte range.'
                ''.format(row + 1, last_byte, field['Byte'], byte_end))

        new_field['byte start'] = byte_start

        # Data type: transform to Numpy format and get shape from its
        # itemsize and the byte range
        dtype = TYPE_MAP_MRC2NPY[field['Data type']]
        byte_size = byte_end - byte_start + 1

        if hasattr(dtype, 'itemsize') and byte_size % dtype.itemsize:
            raise ValueError(
                'in row {}: byte range {} not a multiple of itemsize {} '
                'of the data type {}.'
                ''.format(row + 1, field['Byte'], dtype.itemsize,
                          field['Data type']))

        new_field['dtype'] = dtype
        new_field['byte size'] = byte_size
        # Assuming 1d arrangement of multiple elements
        # TODO: find way to handle 2d fields
        if nend:
            new_field['dshape'] = (nend - nstart + 1,)
        elif hasattr(dtype, 'itemsize'):
            new_field['dshape'] = (byte_size / dtype.itemsize,)
        else:
            new_field['dshape'] = (1,)

        conv_list.append(new_field)

    return tuple(conv_list)


# TODO: make a parent class that just reads the header straight off,
# and make the MRC reader a subclass.
class MRCFileReader(object):

    """Reader for uncompressed binary files including header."""

    def __init__(self, file, header=None, header_fields=None):
        """Initialize a new instance.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in 'rb' mode.
        header : dict, optional
            Dictionary representing the header for the MRC file.
            See `read_header` for a format description.
        header_fields : sequence of dicts, optional
            Definition of the fields in the header (per row), each
            containing the following key-value pairs:

            'name' : label for the field

            'offset' : start of the field in bytes

            'dtype' : data type in Numpy- or Numpy-readable format

            'shape' (optional) : if more than a single entry is read,
            a shape determines the amount read

            If this option is not specified, the MRC2104 specification
            is used.
        """
        try:
            f = open(file, 'rb')
        except TypeError:
            f = file

        if f.mode != 'rb':
            raise ValueError("`file` must be opened in 'rb' mode, got '{}'"
                             "".format(f.mode))

        self.file = f

        # Initialize some attributes to default values
        self.data_shape = None
        self.cell_sides_angstrom = None
        self.mrc_version = None
        self.header_bytes = MRC_HEADER_BYTES

        # Read header, setting above attributes to values
        if header_fields is None:
            self.header_fields = mrc_standardized_fields(
                mrc_spec_fields(MRC_2014_SPEC))
        else:
            self.header_fields = header_fields

        if header is None:
            self.header = self.read_header()
        else:
            self.header = header

        self.data = None

    @classmethod
    def from_raw_file(cls, file, header_bytes, dtype):
        """Construct a reader from a raw file w/o header spec.

        Readers constructed with this method can use `read_data`, but
        not `read_header`.

        Parameters
        ----------
        file : file-like or str
            Stream or filename from which to read the data. The stream
            is allowed to be already opened in 'rb' mode.
        header_bytes : int
            Size of the header in bytes.
        dtype :
            Data type specifier for the data field. It must be
            understood by the `numpy.dtype` constructor.

        Returns
        -------
        reader : `MRCFileReader`
            Raw reader for the given MRC file.
        """
        try:
            f = open(file, 'rb')
        except TypeError:
            f = file

        if f.mode != 'rb':
            raise ValueError("`file` must be opened in 'rb' mode, got '{}'"
                             "".format(f.mode))

        header_bytes = int(header_bytes)
        if header_bytes < 0:
            raise ValueError('`header_bytes` must be nonnegative, got {}.'
                             ''.format(header_bytes))

        filesize_bytes = f.seek(0, 2)
        data_bytes = filesize_bytes - header_bytes
        f.seek(0)

        if header_bytes >= filesize_bytes:
            raise ValueError('`header_bytes` is larger or equal to file size '
                             '({} >= {})'.format(header_bytes, filesize_bytes))

        if dtype is None:
            raise TypeError('`dtype` cannot be `None`')
        dtype = np.dtype(dtype)
        data_size = data_bytes / dtype.itemsize

        instance = cls(f)
        instance.data_dtype = dtype
        instance.data_shape = (data_size,)
        instance.header_bytes = header_bytes
        instance.header = {}
        return instance

    def read_header(self):
        """Read the header from the reader's file.

        The header is also stored in the ``self.header`` attribute.

        Returns
        -------
        header : dict
            Header of ``self.file`` stored in a dictionary, where each
            entry has the following form::

                'name': {'value': value, 'description': description}

            All ``value``'s are `numpy.ndarray`'s with at least one
            dimension. If a ``shape`` is given in ``self.header_fields``,
            the resulting array is reshaped accordingly.
        """
        # First determine the endianness of the data (swapped or not)
        # TODO: determine how to do this

        # Read all fields except data
        header = {}
        for field in self.header_fields:
            # Get all the values from the dictionary
            name = field['name']
            if name == 'data':
                continue
            entry = {'description': field['description']}
            offset_bytes = field['byte start']
            size_bytes = field.get('byte size', None)
            dtype = field['dtype']
            shape = field.get('dshape', None)
            if shape is None:
                shape = -1  # Causes reshape to 1d or no-op

            bytes_per_elem = dtype.itemsize

            if size_bytes is None:
                # Default if 'byte size' is omitted
                num_elems = 1
            else:
                num_elems = size_bytes / bytes_per_elem

            if num_elems != int(num_elems):
                raise RuntimeError(
                    "field '{}': `byte size` {} and `dtype.itemsize` {} "
                    " result in non-integer number of elements"
                    "".format(name, size_bytes, bytes_per_elem))

            # Create format string for struct module to unpack the binary
            # data
            fmt = str(int(num_elems)) + DTYPE_MAP_NPY2STRUCT[dtype]
            if struct.calcsize(fmt) != size_bytes:
                raise RuntimeError(
                    "field '{}': format '{}' has results in {} bytes, but "
                    "`byte size` is {}"
                    "".format(name, fmt, struct.calcsize(fmt), size_bytes))

            self.file.seek(offset_bytes)
            packed_value = self.file.read(size_bytes)
            value = np.array(struct.unpack_from(fmt, packed_value),
                             dtype=dtype)

            if dtype == np.dtype('S1'):
                entry['value'] = ''.join(value.astype(str)).ljust(size_bytes)
            else:
                entry['value'] = value.reshape(shape)
            header[name] = entry

        # Store information gained from the header
        self.header = header
        self._set_attrs_from_header()

        return header

    def _set_attrs_from_header(self):
        """Set the following attributes of ``self`` from ``self.header``:

            - ``data_shape`` : Shape of the (full) data.
            - ``data_dtype`` : Data type of the data.
            - ``cell_sides`` : Size of the unit cell in meters.
            - ``mrc_version`` : ``(major, minor)`` tuple encoding the version
              of the MRC specification used to create the file.
        """
        # data_shape
        nx = self.header['nx']['value']
        ny = self.header['ny']['value']
        nz = self.header['nz']['value']
        self.data_shape = tuple(np.array([nx, ny, nz]).squeeze())

        # data_dtype
        mode = int(self.header['mode']['value'])
        try:
            self.data_dtype = MODE2DTYPE[mode]
        except KeyError:
            raise ValueError('data mode {} not supported'.format(mode))

        # cell_sides
        self.cell_sides_angstrom = np.asarray(self.header['cella']['value'],
                                              dtype=float)
        self.cell_sides = self.cell_sides_angstrom * ANGSTROM

        # mrc_version
        nversion = self.header['nversion']['value']
        maj_ver, min_ver = nversion // 10, nversion % 10
        self.mrc_version = (maj_ver, min_ver)

        # header_bytes
        extra_header_bytes = self.header['nsymbt']['value']
        self.header_bytes = MRC_HEADER_BYTES + extra_header_bytes

    # TODO: read extended header for the standard flavors, see the spec
    # homepage

    def read_data(self, dstart=None, dend=None, store=False):
        """Read the data from the reader's file and store if desired.

        Parameters
        ----------
        dstart : int, optional
            Offset in bytes of the data field. By default, it is equal
            to ``header_size``. Negative values are added to the file
            size in bytes, to support indexing "backwards".
            Use a value different from ``header_size`` to extract data
            subsets.
        dend : `int, optional`
            End position in bytes until which data is read (exclusive).
            Negative values are added to the file size in bytes, to support
            indexing "backwards". Use a value different from the file size
            to extract data subsets.
        store : bool, optional
            If ``True``, store the data in ``self.data`` after reading.

        Returns
        -------
        data : `numpy.ndarray`
            The data read from ``self.file``.
        """
        filesize_bytes = self.file.seek(0, 2)
        if dstart is None:
            dstart_abs = int(self.header_bytes)
        elif dstart < 0:
            dstart_abs = filesize_bytes + int(dstart)
        else:
            dstart_abs = int(dstart) + self.header_bytes

        if dend is None:
            dend_abs = int(filesize_bytes)
        elif dend < 0:
            dend_abs = int(dend) + filesize_bytes
        else:
            dend_abs = int(dend) + self.header_bytes

        if dstart_abs >= dend_abs:
            raise ValueError('invalid `dstart` and `dend`, resulting in '
                             'absolute `dstart` >= `dend` ({} >= {})'
                             ''.format(dstart_abs, dend_abs))
        if dstart_abs < self.header_bytes:
            raise ValueError('invalid `dstart`, resulting in absolute '
                             '`dstart` < `header_bytes` ({} < {})'
                             ''.format(dstart_abs, self.header_bytes))
        if dend_abs > filesize_bytes:
            raise ValueError('invalid `dend`, resulting in absolute '
                             '`dend` > `filesize_bytes` ({} < {})'
                             ''.format(dend_abs, filesize_bytes))

        num_elems = (dend_abs - dstart_abs) / self.data_dtype.itemsize
        if num_elems != int(num_elems):
            raise ValueError('trying to read {} bytes, corresponding to '
                             '{} elements of type {}'
                             ''.format(dend_abs - dstart_abs, num_elems,
                                       self.data_dtype))
        self.file.seek(dstart_abs)
        # TODO: use byteorder according to header
        data = np.fromfile(self.file, dtype=self.data_dtype,
                           count=int(num_elems))

        if dstart_abs == self.header_bytes and dend_abs == filesize_bytes:
            # Full dataset read, reshape to stored shape.
            # Use 'F' order in reshaping since it's the native MRC data
            # ordering.
            data = data.reshape(self.data_shape, order='F')

        if store:
            self.data = data

        return data
