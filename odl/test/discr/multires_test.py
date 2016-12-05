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

"""Unit tests for `multires`."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import numpy as np

import odl
from odl.discr.multires import reduce_over_partition
from odl.util.testutils import (
    all_almost_equal, all_equal, almost_equal, simple_fixture)


# np.average does not have an `out` argument so we test that case with it
reduction = simple_fixture('reduction',
                           params=[np.sum, np.mean, np.average],
                           fmt=' {name} = np.{value.__name__} ')
pad_const = simple_fixture('pad_const', params=[0.0, -1.0])


def test_reduce_over_partition_partial_coverage(reduction, pad_const):
    """Check result against simple impl for partially covering domains."""
    part = odl.uniform_partition([0, 0], [1, 2], (4, 4))
    # Covers the inner 2x2 block of the 4x4 partition
    space = odl.uniform_discr([0.25, 0.5], [0.75, 1.5], (10, 10))
    func = space.one()

    result = reduce_over_partition(func, part, reduction, pad_const)
    expected_result = np.full(part.shape, pad_const)
    expected_result[1:-1, 1:-1] = reduction(func.asarray().reshape([2, 2, -1]),
                                            axis=-1)
    assert all_almost_equal(result, expected_result)


def test_reduce_over_partition_full_coverage(reduction):
    """Check result against simple impl for fully covering domains."""
    part = odl.uniform_partition([0, 0], [1, 2], (4, 4))
    space = odl.uniform_discr([0, 0], [1, 2], (20, 20))
    func = space.one()

    result = reduce_over_partition(func, part, reduction=reduction)
    expected_result = reduction(func.asarray().reshape(part.shape + (-1,)),
                                axis=-1)
    assert all_almost_equal(result, expected_result)


def test_reduce_over_partition_explicit():
    """Check computational result against a hand-calculated example."""
    # Spatial arrangement:
    #
    # Axis 0:
    # - left boundaries coincide
    # - fine grid extends over exactly 3.5 coarse cells
    #
    # Axis 1:
    # - left boundary lies 0.1 into the second coarse cell, which is 10 small
    #   cells
    # - right boundaries coincide
    #
    # A full coarse cell contains 10 * 40 = 400 small cells, hence the
    # fully contained cells should give a sum of 400.
    # In axis 0, the last cell is contained half, hence its contribution is
    # halved.
    # In axis 1, the second coarse cell contains only 30 of 40 small cells,
    # which amounts to a factor of 3/4 there. The first 3 along axis 0
    # thus give value 300, the last one half of it, 150.
    part = odl.uniform_partition([0, 0], [1, 2], (10, 5))
    space = odl.uniform_discr([0.0, 0.5], [0.35, 2.0], (35, 150))
    func = space.one()
    expected_result = np.array(
        [[0, 300, 400, 400, 400],
         [0, 300, 400, 400, 400],
         [0, 300, 400, 400, 400],
         [0, 150, 200, 200, 200],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])

    result = reduce_over_partition(func, part, reduction=np.sum)
    assert all_almost_equal(result, expected_result)

    out = np.empty(part.shape)
    reduce_over_partition(func, part, reduction=np.sum, out=out)
    assert all_almost_equal(out, expected_result)


def test_reduce_over_partition_raise():
    """Check that exceptions are raised under certain error conditions."""
    part = odl.uniform_partition([0, 0], [1, 2], (4, 4))
    # Covers the inner 2x2 block of the 4x4 partition
    space = odl.uniform_discr([0.25, 0.5], [0.75, 1.5], (10, 10))
    func = space.one()

    # Bogus input parameters
    with pytest.raises(TypeError):
        reduce_over_partition(func.asarray(), part, np.sum)
    with pytest.raises(TypeError):
        reduce_over_partition(func, part.grid, np.sum)
    with pytest.raises(TypeError):
        reduce_over_partition(func, part, np.sum, out=func)

    # Non-castable data types
    out = np.empty(part.shape, dtype=int)
    with pytest.raises(ValueError):
        reduce_over_partition(func, part, np.sum, out=out)
    with pytest.raises(ValueError):
        reduce_over_partition(func, part, np.sum, pad_const=1j)

    # Non-uniform stuff
    nonuni_part = odl.nonuniform_partition([0, 1, 2, 3], [0, 1, 2])
    with pytest.raises(ValueError):
        reduce_over_partition(func, nonuni_part, np.sum)

    # Partition too small for space
    small_part = odl.uniform_partition([0, 0], [1, 1], (4, 4))
    with pytest.raises(ValueError):
        reduce_over_partition(func, small_part, np.sum)

    # Bad output shape
    out = np.empty((10, 10))
    with pytest.raises(ValueError):
        reduce_over_partition(func, part, np.sum, out=out)

    # Cell side ratio or relative shift non-integer multiple of space cell
    # sides
    bad_space1 = odl.uniform_discr([0.25, 0.5], [0.75, 1.5], (11, 11))
    with pytest.raises(ValueError):
        reduce_over_partition(bad_space1.one(), part, np.sum)

    bad_space2 = odl.uniform_discr([0.22, 0.5], [0.72, 1.5], (10, 10))
    with pytest.raises(ValueError):
        reduce_over_partition(bad_space2.one(), part, np.sum)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
