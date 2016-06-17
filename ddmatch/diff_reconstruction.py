# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:40:01 2016

@author: chong
"""

# Initial setup
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
import numexpr
import numba
import matplotlib.pyplot as plt
import time
import os
import ddmatch
import odl
import scipy.ndimage as ndimage
from matplotlib.transforms import Bbox


def plot_warp(xphi, yphi, downsample='auto', **kwarg):
    if (downsample == 'auto'):
        skip = np.max([xphi.shape[0]/32, 1])
    elif (downsample == 'no'):
        skip = 1
    else:
        skip = downsample
    plt.plot(xphi[::skip, ::skip], yphi[::skip, ::skip], 'k',
             linewidth=0.5, **kwarg)
    plt.plot(xphi[::skip, ::skip].T, yphi[::skip, ::skip].T, 'k',
             linewidth=0.5, **kwarg)


def get_dir_name(I0name, I1name, sigma):
    file_dir, file_name0 = os.path.split(I0name)
    file_dir, file_name1 = os.path.split(I1name)
    dir_name = os.path.join(file_dir,
                            os.path.splitext(file_name1)[0] +
                            ' to ' + os.path.splitext(file_name0)[0] +
                            ' with sigma ' + str(sigma))
    return dir_name


#I0name = 'Example3 letters/c_highres.png'
#I1name = 'Example3 letters/i_highres.png'
# I0name = 'Example3 letters/eight.png'
# I1name = 'Example3 letters/b.png'
I0name = 'Example3 letters/v.png'
I1name = 'Example3 letters/j.png'
#I0name = 'Example9 letters big/V.png'
#I1name = 'Example9 letters big/J.png'
#I0name = 'Example11 skulls/handnew1.png'
#I1name = 'Example11 skulls/handnew2.png'
# I0name = 'Example8 brains/DS0002AxialSlice80.png'
# I1name = 'Example8 brains/DS0003AxialSlice80.png'

sigma = 1e-1
epsilon = 0.2

sigma = 0.05
epsilon = 0.2
n_iter = 400

I0 = plt.imread(I0name).astype('float')
I1 = plt.imread(I1name).astype('float')

# Apply Gaussian filter
# I0 = ndimage.gaussian_filter(I0, sigma=2)
# I1 = ndimage.gaussian_filter(I1, sigma=2)

dm = ddmatch.TwoComponentDensityMatching(source=I1, target=I0, sigma=sigma)

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

dm.run(n_iter, epsilon=epsilon)

W_square = dm.W**2

plt.figure(1, figsize=(11.7, 9))
plt.clf()

plt.subplot(2, 2, 1)
plt.imshow(dm.I0, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
plt.colorbar()
plt.title('Ground truth')

plt.subplot(2, 2, 2)
plt.imshow(dm.I1, cmap='bone', vmin=dm.I1.min(), vmax=dm.I1.max())
plt.colorbar()
plt.title('Template')

plt.subplot(2, 2, 3)
plt.imshow(W_square, cmap='bone', vmin=W_square.min(), vmax=W_square.max())
plt.colorbar()
plt.title('Warped image')

jac_ax = plt.subplot(2, 2, 4)
mycmap = 'PiYG'
# mycmap = 'Spectral'
# mycmap = 'PRGn'
# mycmap = 'BrBG'
plt.imshow(dm.J, cmap=mycmap, vmin=dm.J.min(), vmax=1.+(1.-dm.J.min()))
plt.gca().set_autoscalex_on(False)
plt.gca().set_autoscaley_on(False)
# plot_warp(dm.phiinvx, dm.phiinvy, downsample=8)
jac_colorbar = plt.colorbar()
plt.title('Jacobian')

# plt.tight_layout()

plt.figure(2, figsize=(7, 7))
plt.clf()
plot_warp(dm.phiinvx, dm.phiinvy, downsample=1)
plt.axis('equal')
warplim = [dm.phiinvx.min(), dm.phiinvx.max(),
           dm.phiinvy.min(), dm.phiinvy.max()]
warplim[0] = min(warplim[0], warplim[2])
warplim[2] = warplim[0]
warplim[1] = max(warplim[1], warplim[3])
warplim[3] = warplim[1]

plt.axis(warplim)
# plt.axis('off')
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Warp')

plt.figure(3, figsize=(8, 1.5))
plt.clf()
plt.plot(dm.E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])


#def full_extent(ax, jac_colorbar, pad=0.0):
#    """Get the full extent of an axes, including axes labels, tick labels, and
#    titles."""
#    # For text objects, we need to draw the figure first, otherwise the extents
#    # are undefined.
#    ax.figure.canvas.draw()
#    items = ax.get_xticklabels() + ax.get_yticklabels()
#    items += jac_colorbar.ax.get_xticklabels() + jac_colorbar.ax.get_yticklabels()
#
#    #    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
#    items += [ax, ax.title, jac_colorbar.ax]
#    bbox = Bbox.union([item.get_window_extent() for item in items])
#
#    return bbox.expanded(1.0 + pad, 1.0 + pad)
#
## Setup directories and files
#fig_dir_name = os.path.join(get_dir_name(I0name, I1name, sigma), 'figures')
#if not os.path.exists(fig_dir_name):
#    os.makedirs(fig_dir_name)
#    print("Creating directory " + fig_dir_name)
#fig = plt.figure(1)
#plt.savefig(os.path.join(fig_dir_name, 'densities.png'),
#            dpi=300, bbox_inches='tight')
#jac_extent = full_extent(jac_ax, jac_colorbar).transformed(fig.dpi_scale_trans.inverted())
#jac_ax.axes.xaxis.set_ticklabels([])
#jac_ax.axes.yaxis.set_ticklabels([])
#jac_ax.set_title('')
#plt.savefig(os.path.join(fig_dir_name, 'jacobian.png'),
#            dpi=150, bbox_inches=jac_extent.expanded(1.02, 1.02))
## plt.savefig(os.path.join(fig_dir_name,'jacobian.pdf'), bbox_inches=jac_extent.expanded(1.02, 1.02))
#
#plt.figure(2)
#plt.axis('off')
#plt.title('')
#plt.savefig(os.path.join(fig_dir_name, 'warp.png'),
#            dpi=150, bbox_inches='tight')
## plt.savefig(os.path.join(fig_dir_name,'warp.pdf'), bbox_inches='tight')
#plt.figure(3)
#plt.savefig(os.path.join(fig_dir_name, 'energy.png'),
#            dpi=150, bbox_inches='tight')
## plt.savefig(os.path.join(fig_dir_name,'energy.pdf'), bbox_inches='tight')
#
#n_anim = 20  # Number of outputs
#anim_indices = np.flipud(np.logspace(np.log10(n_iter), 0 , n_anim)).astype('int')
#anim_slice = np.diff(anim_indices)
#anim_slice = np.where(anim_slice==0, 1, anim_slice)
#anim_indices = np.cumsum(anim_slice)
#
## Setup directories and files
#anim_dir_name = os.path.join(get_dir_name(I0name, I1name, sigma), 'anim')
#if not os.path.exists(anim_dir_name):
#    os.makedirs(anim_dir_name)
#    print("Creating directory " + anim_dir_name)
#anim_basename = os.path.splitext(os.path.basename(I1name))[0] + '_to_' + os.path.splitext(os.path.basename(I0name))[0]
#
## Carry out the simulation and save animation output
#dm = ddmatch.TwoComponentDensityMatching(source=I1, target=I0, sigma=sigma)
#plt.imsave(os.path.join(anim_dir_name, anim_basename + "_target"%k +'.png'), dm.I0, vmin=0., vmax=1., cmap='bone')
#plt.imsave(os.path.join(anim_dir_name, anim_basename + "_source"%k +'.png'), dm.I1, vmin=0., vmax=1., cmap='bone')
#for (nit,k) in zip(anim_slice,np.arange(len(anim_slice))):
#    dm.run(nit, epsilon=epsilon)
#    plt.imsave(os.path.join(anim_dir_name, anim_basename + "_%04.0f"%k +'.png'), dm.W**2, vmin=0., vmax=1., cmap='bone')
