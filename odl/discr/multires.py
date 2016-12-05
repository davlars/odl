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

"""Multiresolution operators and methods."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
import numpy as np

from odl.discr.lp_discr import DiscreteLpElement
from odl.discr.partition import RectPartition
from odl.operator import Operator
from odl.util import writable_array, dtype_repr
from odl.util.numerics import apply_on_boundary


__all__ = ('MaskingOperator', 'show_extent', 'show_both',
           'reduce_over_partition')


class MaskingOperator(Operator):
    """An operator that masks a given spatial region.

    Notes
    -----
    A masking operator :math:`M` for a region-of-interest (ROI),
    applied to a function :math:`f`, returns the function :math:`M(f)`
    given by

    .. math::

        M(f)(x) =
        \\begin{cases}
            0    & \\text{if } x \in \\text{ROI} \\\\
            f(x) & \\text{otherwise.}
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
        This operator sets the region between ``min_pt`` and ``max_pt`` to 0
        scales the contribution from the overlap according to its size.
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


def _apply_reduction(arr, out, reduction, axes):
    try:
        reduction(arr, axis=axes, out=out)
    except TypeError:
        out[:] = reduction(arr, axis=axes)


def reduce_over_partition(discr_func, partition, reduction, pad_const=0,
                          out=None):
    """Reduce a discrete function blockwise over a coarser partition.

    This helper function is intended as a helper for multi-grid
    computations where a finely discretized function needs to undergo
    a blockwise reduction operation over a coarser partition of a
    containing spatial region. An example is to average the given
    function over larger blocks as defined by the partition.

    Parameters
    ----------
    discr_func : `DiscreteLpElement`
        Element in a uniformly discretized function space that is to be
        reduced over blocks defined by ``partition``.
    partition : uniform `RectPartition`
        Coarser partition than ``discr_func.space.partition`` that defines
        the large cells (blocks) over which ``discr_func`` is reduced.
        Its ``cell_sides`` must be an integer multiple of
        ``discr_func.space.cell_sides``.
    reduction : callable
        Reduction function defining the operation on each block of values
        in ``discr_func``. It needs to be callable as
        ``reduction(array, axes=my_axes)`` or
        ``reduction(array, axes=my_axes, out=out_array)``, where
        ``array, out_array`` are `numpy.ndarray`'s, and ``my_axes`` are
        sequence of ints specifying over which axes is being reduced.
        The typical examples are NumPy reductions like `np.sum` or `np.mean`,
        but custom functions are also possible.
    pad_const : scalar, optional
        This value is filled into the parts that are not covered by the
        function.
    out : `numpy.ndarray`, optional
        Array to which the output is written. It needs to have the same
        ``shape`` as ``partition`` and a ``dtype`` to which
        ``discr_func.dtype`` can be cast.

    Returns
    -------
    out : `numpy.ndarray`
        Array holding the result of the reduction operation. If ``out``
        was given, the returned object is a reference to it.
    """
    if not isinstance(discr_func, DiscreteLpElement):
        raise TypeError('`discr_func` must be a `DiscreteLpElement` instance, '
                        'got {!r}'.format(discr_func))
    if not discr_func.space.is_uniform:
        raise ValueError('`discr_func.space` is not uniformly discretized')
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` must be a `RectPartition` instance, '
                        'got {!r}'.format(partition))
    if not partition.is_uniform:
        raise ValueError('`partition` is not uniform')

    # TODO: use different eps in each axis?
    dom_eps = 1e-8 * max(discr_func.space.partition.extent())
    if not partition.set.contains_set(discr_func.space.domain, atol=dom_eps):
        raise ValueError('`partition.set` {} does not contain '
                         '`discr_func.space.domain` {}'
                         ''.format(partition.set, discr_func.space.domain))

    if out is None:
        out = np.empty(partition.shape, dtype=discr_func.dtype,
                       order=discr_func.dtype)
    if not isinstance(out, np.ndarray):
        raise TypeError('`out` must be a `numpy.ndarray` instance, got '
                        '{!r}'.format(out))
    if not np.can_cast(discr_func.dtype, out.dtype):
        raise ValueError('cannot safely cast from `discr_func.dtype` {} '
                         'to `out.dtype` {}'
                         ''.format(dtype_repr(discr_func.dtype),
                                   dtype_repr(out.dtype)))
    if not np.array_equal(out.shape, partition.shape):
        raise ValueError('`out.shape` differs from `partition.shape` '
                         '({} != {})'.format(out.shape, partition.shape))
    if not np.can_cast(pad_const, out.dtype):
        raise ValueError('cannot safely cast `pad_const` {} '
                         'to `out.dtype` {}'
                         ''.format(pad_const, dtype_repr(out.dtype)))
    out.fill(pad_const)

    # Some abbreviations for easier notation
    # All variables starting with "s" refer to properties of
    # `discr_func.space`, whereas "p" quantities refer to the (coarse)
    # `partition`.
    spc = discr_func.space
    smin, smax = spc.min_pt, spc.max_pt
    scsides = spc.cell_sides
    part = partition
    pmin = part.min_pt, part.max_pt
    func_arr = discr_func.asarray()
    ndim = spc.ndim

    # Partition cell sides must be an integer multiple of space cell sides
    csides_ratio_f = part.cell_sides / spc.cell_sides
    csides_ratio = np.around(csides_ratio_f).astype(int)
    if not np.allclose(csides_ratio_f, csides_ratio):
        raise ValueError('`partition.cell_sides` is a non-integer multiple '
                         '({}) of `discr_func.space.cell_sides'
                         ''.format(csides_ratio_f))

    # Shift must be an integer multiple of space cell sides
    rel_shift_f = (smin - pmin) / scsides
    if not np.allclose(np.round(rel_shift_f), rel_shift_f):
        raise ValueError('shift between `partition` and `discr_func.space` '
                         'is a non-integer multiple ({}) of '
                         '`discr_func.space.cell_sides'
                         ''.format(rel_shift_f))

    # Calculate relative position of a number of interesting points

    # Positions of the space domain min and max vectors relative to the
    # partition
    cvecs = part.cell_boundary_vecs
    smin_idx = part.index(smin)
    smin_partpt = np.array([cvec[si + 1] for si, cvec in zip(smin_idx, cvecs)])
    smax_idx = part.index(smax)
    smax_partpt = np.array([cvec[si] for si, cvec in zip(smax_idx, cvecs)])

    # Inner part of the partition in the space domain, i.e. partition cells
    # that are completely contained in the spatial domain and do not touch
    # its boundary
    p_inner_slc = [slice(li + 1, ri) for li, ri in zip(smin_idx, smax_idx)]

    # Positions of the first and last partition points that still lie in
    # the spatial domain, relative to the space partition
    pl_idx = np.round(spc.index(smin_partpt, floating=True)).astype(int)
    pr_idx = np.round(spc.index(smax_partpt, floating=True)).astype(int)
    s_inner_slc = [slice(li, ri) for li, ri in zip(pl_idx, pr_idx)]

    # Slices to constrain to left and right boundary in each axis
    pl_slc = [slice(li, li + 1) for li in smin_idx]
    pr_slc = [slice(ri, ri + 1) for ri in smax_idx]

    # Slices for the overlapping space cells to the left and the right
    # (up to left index excl. / from right index incl.)
    sl_slc = [slice(None, li) for li in pl_idx]
    sr_slc = [slice(ri, None) for ri in pr_idx]

    # Shapes for reduction of the inner part by summing over axes.
    reduce_inner_shape = []
    reduce_axes = tuple(2 * i + 1 for i in range(ndim))
    inner_shape = func_arr[s_inner_slc].shape
    for n, k in zip(inner_shape, csides_ratio):
        reduce_inner_shape.extend([n // k, k])

    # Now we loop over boundary parts of all dimensions from 0 to ndim-1.
    # They are encoded as follows:
    # - We select inner (1) and outer (2) parts per axis by looping over
    #   `product([1, 2], repeat=ndim)`, using the name `parts`.
    # - Wherever there is a 2 in the sequence, 2 slices must be generated,
    #   one for left and one for right. The total number of slices is the
    #   product of the numbers in `parts`, i.e. `num_slcs = prod(parts)`.
    # - We get the indices of the 2's in the sequence and put them in
    #   `outer_indcs`.
    # - The "p" and "s" slice lists are initialized with the inner parts.
    #   We need `num_slcs` such lists for this particular sequence `parts`.
    # - Now we enumerate `outer_indcs` as `i, oi` and put into the
    #   (2*i)-th entry of the slice lists the "left" outer slice and into
    #   the (2*i+1)-th entry the "right" outer slice.
    #
    # The total number of slices to loop over is equal to
    # sum(k=0->ndim, binom(ndim, k) * 2^k) = 3^ndim.
    # This should not add too much computational overhead.
    for parts in product([1, 2], repeat=ndim):

        # Number of slices to consider
        num_slcs = np.prod(parts)

        # Indices where we need to consider the outer parts
        outer_indcs = tuple(np.where(np.equal(parts, 2))[0])

        # Initialize the "p" and "s" slice lists with the inner slices.
        # Each list contains `num_slcs` of those.
        p_slcs = [list(p_inner_slc) for _ in range(num_slcs)]
        s_slcs = [list(s_inner_slc) for _ in range(num_slcs)]
        # Put the left/right slice in the even/odd sublists at the
        # position indexed by the outer_indcs thing.
        # We also need to initialize the `reduce_shape`'s for all cases,
        # which has the value (n // k, k) for the "inner" axes and
        # (1, n) in the "outer" axes.
        reduce_shapes = [list(reduce_inner_shape) for _ in range(num_slcs)]
        for islc, bdry in enumerate(product('lr', repeat=len(outer_indcs))):
            for oi, l_or_r in zip(outer_indcs, bdry):
                if l_or_r == 'l':
                    p_slcs[islc][oi] = pl_slc[oi]
                    s_slcs[islc][oi] = sl_slc[oi]
                else:
                    p_slcs[islc][oi] = pr_slc[oi]
                    s_slcs[islc][oi] = sr_slc[oi]

            f_view = func_arr[s_slcs[islc]]
            for oi in outer_indcs:
                reduce_shapes[islc][2 * oi] = 1
                reduce_shapes[islc][2 * oi + 1] = f_view.shape[oi]

        # Compute the block reduction of all views represented by the current
        # `parts`. This is done by reshaping from the original shape to the
        # above calculated `reduce_shapes` and reducing over `reduce_axes`.
        for p_s, s_s, red_shp in zip(p_slcs, s_slcs, reduce_shapes):
            f_view = func_arr[s_s]
            out_view = out[p_s]

            if 0 not in f_view.shape:
                # View not empty, reduction makes sense
                _apply_reduction(arr=f_view.reshape(red_shp), out=out_view,
                                 axes=reduce_axes, reduction=reduction)
    return out


# --- Tomography-specific stuff --- #

# TODO: move to some location in odl/tomo


def extent(angle, corners, detector_pos):
    """
    Compute the detector extent of a masking region for
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


def show_extent(data, min_pt, max_pt, detector_pos):
    corners = [[min_pt[0], max_pt[1]],
               [min_pt[0], min_pt[1]],
               [max_pt[0], min_pt[1]],
               [max_pt[0], max_pt[1]]]

    """Show the sinogram data along with the mask extent"""
    fig, ax = plt.subplots()

    xrange = [data.space.min_pt[0], data.space.max_pt[0]]
    yrange = [data.space.min_pt[1], data.space.max_pt[1]]

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.imshow(np.rot90(data), extent=[*xrange, *yrange], cmap='bone')
    ax.set_aspect('auto', 'box')

    # TODO: generalize
    thetas = np.linspace(data.space.min_pt[0], data.space.max_pt[0],
                         data.shape[0], endpoint=False)

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
        bbox_image.set_data(np.rot90(data.asarray(), -1))
        ax.add_artist(bbox_image)

    show(coarse_data)
    show(fine_data)

    ax.set_aspect('auto', 'box')

# TODO:
# - Add a 'multi-grid' space, which can be used for reconstruction and so on
# - Add support multi-resolution phantoms
# - Define definitive API for multi-grid reconstruction
