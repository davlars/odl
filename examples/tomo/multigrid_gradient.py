#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:58:10 2016

@author: hkohr
"""

import numpy as np
import odl
from odl.util import writable_array
from builtins import super


# %%

class MaskingOperator(odl.Operator):

    def __init__(self, space, min_pt, max_pt):

        super().__init__(domain=space, range=space, linear=True)
        # TODO: allow ignoring axes
        # TODO: allow masking array
        # TODO: sanitize input
        self.min_pt = min_pt
        self.max_pt = max_pt

    def _call(self, x, out):

        idx_min = self.domain.index(self.min_pt)
        idx_max = self.domain.index(self.max_pt)

        slc = tuple(slice(imin, imax) for imin, imax in zip(idx_min, idx_max))
        out.assign(x)
        with writable_array(out) as out_arr:
            out_arr[slc] = 0

    @property
    def adjoint(self):
        return self


# %% MultigridGradient

class MultiGridGradient(odl.DiagonalOperator):

    def __init__(self, coarse_discr, fine_discr, method='forward',
                 pad_mode='constant', pad_const=0):

        self.coarse_grad = odl.Gradient(coarse_discr, method=method,
                                        pad_mode=pad_mode, pad_const=pad_const)

        # TODO: handle case when fine discr touches boundary
        # TODO: change pad_mode, this is just for debugging
        self.fine_grad = odl.Gradient(fine_discr, method=method,
                                      pad_mode='constant', pad_const=0)

        super().__init__(self.coarse_grad, self.fine_grad)

    @property
    def coarse_discr(self):
        return self.domain[0]

    @property
    def fine_discr(self):
        return self.domain[1]

    def overlaps(self, coarse_func, fine_func):
        # TODO: generalize to non-matching overlaps
        # TODO: handle corner cases of small inserts etc.
        # TODO: clean up notation regarding min<->left, max<->right
        # TODO: shorten names without losing clarity

        # TODO: turn into proper error messages & put checks into __init__
        assert isinstance(coarse_func, odl.DiscreteLpElement)
        assert isinstance(fine_func, odl.DiscreteLpElement)
        assert coarse_func.ndim == fine_func.ndim
        assert np.all(coarse_func.space.min_pt <= fine_func.space.min_pt)
        assert np.all(coarse_func.space.max_pt >= fine_func.space.max_pt)

        ndim = self.coarse_discr.ndim

        coarse_cell_sides = self.coarse_discr.cell_sides
        coarse_shape = self.coarse_discr.shape
        fine_cell_sides = self.fine_discr.cell_sides
        yleft, yright = self.fine_discr.min_pt, self.fine_discr.max_pt

        # Get indices of the minimum and maximum coordinates of the insert
        yleft_idcs = self.coarse_discr.index(yleft)
        yleft_idcs_f = self.coarse_discr.index(yleft, floating=True)
        yright_idcs = self.coarse_discr.index(yright)
        yright_idcs_f = self.coarse_discr.index(yright, floating=True)

        # TODO: relax this restriction
        assert np.all(yleft_idcs > 0)
        assert np.all(yleft_idcs < np.array(coarse_shape) - 1)
        assert np.all(yright_idcs > 0)
        assert np.all(yright_idcs < np.array(coarse_shape) - 1)

        print('yleft:', yleft)
        print('yleft_idcs:', yleft_idcs)
        print('yleft_idcs_f:', yleft_idcs_f)
        print('yright:', yright)
        print('yright_idcs:', yright_idcs)
        print('yright_idcs_f:', yright_idcs_f)

        # TODO: handle case of first/last grid points
        # Right now we're protected by the index condition further up
        coarse_gvecs = self.coarse_discr.grid.coord_vectors
        left_gridpt = [gvec[li]
                       for gvec, li in zip(coarse_gvecs, yleft_idcs)]
        left_gridpt_before = [gvec[li - 1]
                              for gvec, li in zip(coarse_gvecs, yleft_idcs)]
        right_gridpt = [gvec[ri]
                        for gvec, ri in zip(coarse_gvecs, yright_idcs)]
        right_gridpt_after = [gvec[ri + 1]
                              for gvec, ri in zip(coarse_gvecs, yright_idcs)]

        print('left_gridpt:', np.array(left_gridpt))
        print('left_gridpt_before:', np.array(left_gridpt_before))
        print('right_gridpt:', np.array(right_gridpt))
        print('right_gridpt_after:', np.array(right_gridpt_after))

        left_fine_parts, right_fine_parts = [], []
        fine_meshgrid = fine_discr.meshgrid
        broadcast_slice = [None] * ndim
        for i in range(ndim):
            bcast_slc = list(broadcast_slice)
            bcast_slc[i] = slice(None)

            # Construct meshgrids with thickness 1 in axis i and interpolate
            # the finely sampled function there.

            # For fine discr, "left before" and "right after"
            fine_mg = list(fine_meshgrid)
            fine_mg[i] = np.atleast_1d(left_gridpt_before[i])[bcast_slc]
            left_fine_parts.append(fine_func.interpolation(fine_mg))
            print('left interp:', left_fine_parts[i])

            fine_mg[i] = np.atleast_1d(right_gridpt_after[i])[bcast_slc]
            right_fine_parts.append(fine_func.interpolation(fine_mg))
            print('right interp:', right_fine_parts[i])

        # Compute the contribution of the fine part overlapping with the
        # interval into which the fine min and max points fall. We first
        # compute the averaged arrays, then create auxiliary spaces with 2
        # intervals in the respective axis, and finally interpolate between
        # the last "regular" value on the coarse grid and the averaged fine
        # values.

        # TODO: take one extra if available
        coarse_cvecs = self.coarse_discr.partition.coord_vectors
        left_cell_bdry_after = [
            cvec[li + 1] for cvec, li in zip(coarse_cvecs, yleft_idcs)]
        left_cell_bdry_after_fine_idcs = self.fine_discr.index(
            left_cell_bdry_after)
        right_cell_bdry_before = [
            cvec[ri - 1] for cvec, ri in zip(coarse_cvecs, yright_idcs)]
        right_cell_bdry_before_fine_idcs = self.fine_discr.index(
            right_cell_bdry_before)

        print('left_cell_bdry_after:', left_cell_bdry_after)
        print('right_cell_bdry_before:', right_cell_bdry_before)
        print('left_cell_bdry_after_fine_idcs:',
              left_cell_bdry_after_fine_idcs)
        print('right_cell_bdry_before_fine_idcs:',
              right_cell_bdry_before_fine_idcs)

        # Compute the mean arrays along all axes, left and right, along with
        # their weights
        left_mean_arrays, right_mean_arrays = [], []
        left_mean_weights, right_mean_weights = [], []
        take_all_slice = [slice(None)] * ndim
        fine_arr = fine_func.asarray()
        for i in range(ndim):
            slc_l = list(take_all_slice)
            slc_l[i] = slice(left_cell_bdry_after_fine_idcs[i])
            left_mean_arrays.append(np.mean(fine_arr[slc_l], axis=i))
            print(left_mean_arrays[i])

            slc_r = list(take_all_slice)
            slc_r[i] = slice(right_cell_bdry_before_fine_idcs[i])
            right_mean_arrays.append(np.mean(fine_arr[slc_r], axis=i))
            print(right_mean_arrays[i])

        # In each axis and for left/right, create auxiliary coarse spaces
        # with size 2 in the respective axis. Create an element from the
        # last available "real" values and the averaged fine values.
        for i in range(ndim):


        left_resampled_parts, right_resampled_parts = [], []
        fine_meshgrid = fine_discr.meshgrid
        for i in range(ndim):
            print('')
            print('axis', i)
            print('')

            # Get the views into the fine array with thickness 1 in axis
            # i, left and right. Append to the respective lists.
            left_fine_slc = list(take_all_slice)
            left_fine_slc[i] = 0
            left_fine_parts.append(fine_arr[tuple(left_fine_slc)])
            right_fine_slc = [slice(None)] * ndim
            right_fine_slc[i] = -1
            right_fine_parts.append(fine_arr[tuple(right_fine_slc)])

            aux_cell_sides = np.copy(fine_cell_sides)
            aux_cell_sides[i] = coarse_cell_sides[i]
            print('aux_cell_sides:', aux_cell_sides)

            left_min_pt = np.copy(yleft)
            left_min_pt[i] = yleft_before[i]
            left_max_pt = np.copy(yright)
            left_max_pt[i] = yleft_after[i]
            aux_space_min = odl.uniform_discr_fromdiscr(
                coarse_discr, min_pt=left_min_pt, max_pt=left_max_pt,
                cell_sides=aux_cell_sides)
            print('left_min_pt:', left_min_pt)
            print('left_max_pt:', left_max_pt)
            print('min space shape:', aux_space_min.shape)
            print('min space min/max:', aux_space_min.min_pt,
                  aux_space_min.max_pt)

            right_min_pt = np.copy(yleft)
            right_min_pt[i] = yright_before[i]
            right_max_pt = np.copy(yright)
            right_max_pt[i] = yright_after[i]
            aux_space_max = odl.uniform_discr_fromdiscr(
                coarse_discr, min_pt=right_min_pt, max_pt=right_max_pt,
                cell_sides=aux_cell_sides)
            print('right_min_pt:', right_min_pt)
            print('right_max_pt:', right_max_pt)
            print('max space shape:', aux_space_max.shape)
            print('max space min/max:', aux_space_max.min_pt,
                  aux_space_max.max_pt)

            mg_slc = [None] * ndim
            mg_slc[i] = slice(None)
            left_meshgrid = list(fine_meshgrid)
            left_meshgrid[i] = np.atleast_1d(yleft[i])[tuple(mg_slc)]
            left_resampled_parts.append(aux_space_min.element(
                coarse_func.interpolation(tuple(left_meshgrid))))

            right_meshgrid = list(fine_meshgrid)
            right_meshgrid[i] = np.atleast_1d(yright[i])[tuple(mg_slc)]
            right_resampled_parts.append(aux_space_max.element(
                coarse_func.interpolation(tuple(right_meshgrid))))

        return (left_fine_parts, right_fine_parts,
                left_resampled_parts, right_resampled_parts)

    def _call(self, x, out):
        pass

    @property
    def adjoint(self):
        return None


# %% Set up the operators and phantoms

coarse_discr = odl.uniform_discr([-10, -10], [10, 10], [50, 50],
                                 dtype='float32')
fine_min = [1, 1]
fine_max = [2, 2]
fine_discr = odl.uniform_discr(fine_min, fine_max, [100, 200], dtype='float32')

multi_grad = MultiGridGradient(coarse_discr, fine_discr)

coarse_func = 2 * coarse_discr.one()
fine_func = fine_discr.one()

multi_grad.overlaps(coarse_func, fine_func)