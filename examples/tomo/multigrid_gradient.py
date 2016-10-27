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

        fine_arr = fine_func.asarray()

        coarse_cell_sides = self.coarse_discr.cell_sides
        fine_cell_sides = self.fine_discr.cell_sides
        xmin, xmax = self.coarse_discr.min_pt, self.coarse_discr.max_pt
        ymin, ymax = self.fine_discr.min_pt, self.fine_discr.max_pt

        ymin_idcs = self.coarse_discr.index(ymin)
        ymin_idcs_f = self.coarse_discr.index(ymin, floating=True)
        ymax_idcs = self.coarse_discr.index(ymax)
        ymax_idcs_f = self.coarse_discr.index(ymax, floating=True)

        print('ymin:', ymin)
        print('ymin_idcs:', ymin_idcs)
        print('ymin_idcs_f:', ymin_idcs_f)
        print('ymax:', ymax)
        print('ymax_idcs:', ymax_idcs)
        print('ymax_idcs_f:', ymax_idcs_f)

        coarse_cvecs = self.coarse_discr.partition.coord_vectors
        ymin_after = tuple(cvec[i + 1]
                           for cvec, i in zip(coarse_cvecs, ymin_idcs))
        ymin_before = tuple(cvec[i]
                            for cvec, i in zip(coarse_cvecs, ymin_idcs))
        ymax_before = tuple(cvec[i - 1]
                            for cvec, i in zip(coarse_cvecs, ymax_idcs))
        ymax_after = tuple(cvec[i]
                           for cvec, i in zip(coarse_cvecs, ymax_idcs))

        print('ymin_before:', np.array(ymin_before))
        print('ymin_after:', np.array(ymin_after))
        print('ymax_before:', np.array(ymax_before))
        print('ymax_after:', np.array(ymax_after))

        # TODO: take one extra if available
        ymin_after_fine_idx = self.fine_discr.index(ymin_after)
        ymax_before_fine_idx = self.fine_discr.index(ymax_before)

        print('ymin_after_fine_idx:', ymin_after_fine_idx)
        print('ymax_before_fine_idx:', ymax_before_fine_idx)

        # Compute the mean arrays along all axes left and right
        left_avg_arrays, right_avg_arrays = [], []
        ndim = self.coarse_discr.ndim
        # TODO: perhaps better to make cut-out spaces and use integration
        # there
        for i in range(ndim):
            slc_l = [slice(None)] * ndim
            slc_l[i] = slice(ymin_after_fine_idx[i])
            left_avg_arrays.append(np.mean(fine_arr[slc_l], axis=i))
            print(left_avg_arrays[-1])

            slc_r = [slice(None)] * ndim
            slc_r[i] = slice(ymax_before_fine_idx[i], None)
            right_avg_arrays.append(np.mean(fine_arr[slc_r], axis=i))
            print(right_avg_arrays[-1])

        # TODO: The lambda stuff should be handled by the coarse resampling,
        # remove if not needed.
        left_lambda, left_coarse_idcs = [], []
        for yi, yi_f in zip(ymin_idcs, ymin_idcs_f):
            if np.isclose(yi, yi_f):
                # Very close to a grid point. We take the whole cell at
                # 1 index to the left, with weight 1.
                left_lambda.append(yi_f - yi + 1)
                left_coarse_idcs.append(yi - 1)
            else:
                # Somewhere in between the cell boundaries, just take the
                # values as-is.
                left_lambda.append(yi_f - yi)
                left_coarse_idcs.append(yi)
        print('left_lambda:', left_lambda)
        print('coarse_min_indcs:', left_coarse_idcs)

        right_lambda, right_coarse_idcs = [], []
        for yi, yi_f in zip(ymax_idcs, ymax_idcs_f):
            if np.isclose(yi, yi_f):
                # Very close to a grid point. We take the whole cell at
                # 1 index to the right, with weight 1.
                right_lambda.append(yi_f - yi + 1)
                right_coarse_idcs.append(yi + 1)
            else:
                # Somewhere in between the cell boundaries, just take the
                # values as-is.
                right_lambda.append(yi_f - yi)
                right_coarse_idcs.append(yi)
        print('right_lambda:', right_lambda)
        print('coarse_max_indcs:', right_coarse_idcs)

        # TODO: do both left and right parts with interpolation!!!
        left_fine_parts, right_fine_parts = [], []
        left_resampled_parts, right_resampled_parts = [], []
        fine_meshgrid = fine_discr.meshgrid
        for i in range(ndim):
            print('')
            print('axis', i)
            print('')

            # Get the views into the fine array with thickness 1 in axis
            # i, left and right. Append to the respective lists.
            left_fine_slc = [slice(None)] * ndim
            left_fine_slc[i] = 0
            left_fine_parts.append(fine_arr[tuple(left_fine_slc)])
            right_fine_slc = [slice(None)] * ndim
            right_fine_slc[i] = -1
            right_fine_parts.append(fine_arr[tuple(right_fine_slc)])

            aux_cell_sides = np.copy(fine_cell_sides)
            aux_cell_sides[i] = coarse_cell_sides[i]
            print('aux_cell_sides:', aux_cell_sides)

            left_min_pt = np.copy(ymin)
            left_min_pt[i] = ymin_before[i]
            left_max_pt = np.copy(ymax)
            left_max_pt[i] = ymin_after[i]
            aux_space_min = odl.uniform_discr_fromdiscr(
                coarse_discr, min_pt=left_min_pt, max_pt=left_max_pt,
                cell_sides=aux_cell_sides)
            print('left_min_pt:', left_min_pt)
            print('left_max_pt:', left_max_pt)
            print('min space shape:', aux_space_min.shape)
            print('min space min/max:', aux_space_min.min_pt,
                  aux_space_min.max_pt)

            right_min_pt = np.copy(ymin)
            right_min_pt[i] = ymax_before[i]
            right_max_pt = np.copy(ymax)
            right_max_pt[i] = ymax_after[i]
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
            left_meshgrid[i] = np.atleast_1d(ymin[i])[tuple(mg_slc)]
            left_resampled_parts.append(aux_space_min.element(
                coarse_func.interpolation(tuple(left_meshgrid))))

            right_meshgrid = list(fine_meshgrid)
            right_meshgrid[i] = np.atleast_1d(ymax[i])[tuple(mg_slc)]
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