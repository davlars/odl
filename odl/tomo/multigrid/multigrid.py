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

"""Multigrid operators and methods."""

from builtins import super

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from odl import Operator
from odl.util import writable_array
from odl.util.numerics import apply_on_boundary


__all__ = ('MaskingOperator', 'show_extent', 'show_both')


class MaskingOperator(Operator):

    """An operator that masks a given region of space. A masking function
    :math:`\mathcal{M}` for a region-of-interest (ROI) applied to a function
    :math:`f`, returns the following function:
        .. math::

            \mathcal{M}f(x) = \begin{cases}
                    0 & \text{if } x \in \text{ ROI}\\
                    f(x) & \text{otherwise}
                \end{cases}
    """

    def __init__(self, space, min_pt, max_pt):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp`
            Discretized space, the domain of the masked function
        min_pt, max_pt:  float or sequence of floats
            Minimum/maximum corners of the masked region.

        Notes
        -----
        This operator sets the region between ``min_pt`` and ``max_pt`` to 0,
        and scales overlapping region in discretization taking into account the
        overlap.
        """
        super().__init__(domain=space, range=space, linear=True)
        self.min_pt = min_pt
        self.max_pt = max_pt

    def _call(self, x, out):
        """Mask ``x`` and store the result in ``out`` if given."""
        # TODO: find better way of getting the indices
        idx_min_flt = ((self.min_pt - self.domain.min_pt) /
                       self.domain.cell_sides)
        idx_max_flt = ((self.max_pt - self.domain.min_pt) /
                       self.domain.cell_sides)

        # to deal with coinciding boundaries we introduce an epsilon tolerance
        epsilon = 1e-6
        idx_min = np.floor(idx_min_flt - epsilon).astype(int)
        idx_max = np.ceil(idx_max_flt + epsilon).astype(int)

        def coeffs(d):
            return (1.0 - (idx_min_flt[d] - idx_min[d]),
                    1.0 - (idx_max[d] - idx_max_flt[d]))

        # we need an extra level of indirection in order to capture `d` inside
        # the lambda expressions
        def fn_pair(d):
            return (lambda x: x * coeffs(d)[0], lambda x: x * coeffs(d)[1])

        boundary_scale_fns = [fn_pair(d) for d in range(x.ndim)]

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        slc_inner = tuple(slice(imin + 1, imax - 1) for imin, imax in
                          zip(idx_min, idx_max))

        out.assign(x)
        mask = np.ones_like(x)
        with writable_array(out) as out_arr:
            mask[slc_inner] = 0
            apply_on_boundary(mask[slc],
                              boundary_scale_fns,
                              only_once=False,
                              out=mask[slc])
            apply_on_boundary(mask[slc],
                              lambda x: 1.0 - x,
                              only_once=True,
                              out=mask[slc])
            out_arr[slc] = mask[slc] * out_arr[slc]

    @property
    def adjoint(self):
        """Returns the (self-adjoint) masking operator."""
        return self


def extent(angle, corners, detector_pos):
    """
    Compute the detecftor extent of a masking region for
    a given angle and detector position, for parallel 2d
    """
    angle = np.pi - angle

    regime = (int)(angle / (0.5 * np.pi))

    left_corner = corners[regime]
    right_corner = corners[(regime + 2) % 4]

    def proj_location(x, d, theta):
        return np.dot([np.sin(theta), -np.cos(theta)], x - d(theta))

    return [proj_location(left_corner, detector_pos, angle),
            proj_location(right_corner, detector_pos, angle)]


def show_extent(data, corners, detector_pos):
    """Show the sinogram data along with the mask extent"""
    fig, ax = plt.subplots()

    xrange = [data.space.min_pt[0], data.space.max_pt[0]]
    yrange = [data.space.min_pt[1], data.space.max_pt[1]]

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.imshow(np.rot90(data), extent=[*xrange, *yrange], cmap='bone')
    ax.set_aspect('auto', 'box')

    thetas = np.arange(data.space.min_pt[0], data.space.max_pt[0],
                       (data.space.max_pt[0] - data.space.min_pt[0]) /
                           data.shape[0])

    alpha = [extent(theta, corners, detector_pos) for theta in thetas]
    ax.plot(thetas, alpha, linewidth=2.0)


def show_both(coarse_data, fine_data):
    """Show the coarse and fine reconstruction in a single image"""
    fig, ax = plt.subplots()

    low = np.min([np.min(coarse_data), np.min(fine_data)])
    high = np.max([np.max(coarse_data), np.max(fine_data)])

    normalization = mpl.colors.Normalize(vmin=low, vmax=high)

    ax.set_xlim(coarse_data.space.domain.min_pt[0],
                coarse_data.space.domain.max_pt[0])
    ax.set_ylim(coarse_data.space.domain.min_pt[1],
                coarse_data.space.domain.max_pt[1])

    def show(data):
        bbox0 = Bbox.from_bounds(*data.space.domain.min_pt,
                                 *(data.space.domain.max_pt -
                                   data.space.domain.min_pt))
        bbox = TransformedBbox(bbox0, ax.transData)
        bbox_image = BboxImage(bbox, norm=normalization, cmap='bone',
                               interpolation='nearest', origin=False)
        bbox_image.set_data(np.rot90(data.asarray(),-1))
        ax.add_artist(bbox_image)

    show(coarse_data)
    show(fine_data)

# TODO:
# - Add a 'multi-grid' space, which can be used for reconstruction and so on
# - Add support multi-resolution phantoms
# - Define definitive API for multi-grid reconstruction
