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

"""
Shape-based reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
from builtins import super
import matplotlib.pyplot as plt
from odl.discr import (Gradient, Divergence, uniform_discr,
                       uniform_partition, ResizingOperator, DiscreteLp)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
from odl.tomo import Parallel2dGeometry, RayTransform, fbp_op
from odl.phantom import (white_noise, disc_phantom, submarine,
                         shepp_logan, geometric)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.deform.linearized import _linear_deform
from odl.deform.mass_preserving import geometric_deform, mass_presv_deform
standard_library.install_aliases()


__all__ = ('LDDMM_gradient_descent_scheme_solver',)


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op


# Kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


def fitting_kernel_ft(kernel):
    """Compute the n-D Fourier transform of the discrete kernel ``K``.

    Calculate the n-D Fourier transform of the discrete kernel ``K`` on the
    image grid points {y_i} to its reciprocal points {xi_i}.

    """
    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return vectorial_ft_fit_op(discretized_kernel)


def shepp_logan_ellipse_2d_template():
    """Return ellipse parameters for a 2d Shepp-Logan phantom.

    This assumes that the ellipses are contained in the square
    [-1, -1]x[-1, -1].
    """
#    return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
#            [-.98, .6624, .8740, 0.0000, -.0184, 0],
#            [-.02, .1100, .3100, 0.2200, 0.0000, -18],
#            [-.02, .1600, .4100, -.2200, 0.0000, 18],
#            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
#            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
#            [0.01, .0460, .0460, 0.0000, -.1000, 0],
#            [0.01, .0460, .0230, -.0800, -.6050, 0],
#            [0.01, .0230, .0230, 0.0000, -.6060, 0],
#            [0.01, .0230, .0460, 0.0600, -.6050, 0]]
    #       value  axisx  axisy     x       y  rotation           
#    # Shepp-Logan region of interest
#    return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
#            [-.98, .6624, .8740, 0.0000, -.0184, 0],
#            [-.02, .1400, .1400, 0.2200, 0.0000, -18],
#            [-.02, .1600, .4100, -.2200, 0.0000, 18],
#            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
#            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
#            [0.01, .0460, .0460, 0.0000, -.1000, 0],
#            [0.01, .0460, .0230, -.0800, -.6050, 0],
#            [0.01, .0230, .0230, 0.0000, -.6060, 0],
#            [0.01, .0230, .0460, 0.0600, -.6050, 0]]
    return [[2.00, .6000, .6000, 0.0000, 0.1200, 0],
            [-.98, .5624, .5640, 0.0000, -.0184 + 0.12, 0],
            [-.02, .1100, .1100, 0.2600, 0.1500, -18],
            [-.02, .1300, .1300, -.2500, 0.2000, 18],
            [0.01, .1650, .1650, 0.0000, 0.3000, 0],
            [0.01, .0300, .0300, 0.0000, 0.1400, 0],
            [0.01, .0300, .0300, -.1400, 0.1000, 0],
            [0.01, .0360, .0230, -.0770, -.2050, 0],
            [0.01, .0230, .0230, 0.0000, -.2060, 0],
            [0.01, .0230, .0360, 0.0600, -.2050, 0]] 

#template = shepp_logan_2d(space, modified=True)
#template.show('template')


def modified_shepp_logan_ellipses(ellipses):
    """Modify ellipses to give the modified Shepp-Logan phantom.

    Works for both 2d and 3d.
    """
    intensities = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    assert len(ellipses) == len(intensities)

    for ellipse, intensity in zip(ellipses, intensities):
        ellipse[0] = intensity


def shepp_logan_ellipses(ndim, modified=False):
    """Ellipses for the standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    ndim : {2, 3}
        Dimension of the space the ellipses should be in.
    modified : bool, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    ellipse_phantom : Function for creating arbitrary ellipse phantoms
    shepp_logan : Create a phantom with these ellipses

    References
    ----------
    .. _Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    if ndim == 2:
        ellipses = shepp_logan_ellipse_2d_template()
    else:
        raise ValueError('dimension not 2, no phantom available')

    if modified:
        modified_shepp_logan_ellipses(ellipses)

    return ellipses


def shepp_logan_2d(space, modified=False):
    """Standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
    modified : `bool`, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    shepp_logan_ellipses : Get the parameters that define this phantom
    ellipse_phantom : Function for creating arbitrary ellipse phantoms

    References
    ----------
    .. Shepp-Logan phantom: en.wikipedia.org/wiki/Shepp–Logan_phantom
    """
    ellipses = shepp_logan_ellipses(space.ndim, modified)

    return geometric.ellipse_phantom(space, ellipses)


def LDDMM_gradient_descent_scheme_solver(gradS, I, time_pts, niter, eps,
                                         lamb, impl='geom', callback=None):
    """
    Solver for the shape-based reconstruction using LDDMM.

    The model is:

    min sigma * (1 - sqrt{DetJacInvPhi})^2 + (T(phi.I) - g)^2,
    where phi.I := DetJacInvPhi * I(InvPhi) is a mass-preserving deformation.

    Note that:
    If T is an identity operator, the above model reduces for image matching.
    If T is a forward projection operator, the above model is
    for image reconstruction.

    Parameters
    ----------
    gradS : `Operator`
        op.adjoint * (op - noise_proj_data),
        where op is a forward operator, noise_proj_data is the given data.
    I : `DiscreteLpElement`
        Fixed template deformed by the deformation.
    time_pts : `int`
        The number of time intervals
    iter : 'int'
        The given maximum iteration number.
    eps : 'float'
        The given step size.
    lamb : 'float'
        The given regularization parameter. It's a wight on 
        regularization-term side.
    impl : 'string'
        The given implementation method for mass preserving or not.
        The impl chooses 'mp' or 'geom', where 'mp' means using
        mass-preserving method, and 'geom' means using
        non-mass-preserving method. Its defalt choice is 'geom'.
    callback : 'Class'
        Show the iterates.
    """

    # Give the number of time intervals
    N = time_pts

    # Give the inverse of time intervals
    inv_N = 1.0 / N

    # Create the space of image
    image_domain = gradS.domain

    # Get the dimansion
    dim = image_domain.ndim

    # Create the space for series deformations and series Jacobian determinant
    pspace = image_domain.tangent_bundle
    series_pspace = ProductSpace(pspace, N+1)
    series_image_space = ProductSpace(image_domain, N+1)

    # Initialize vector fileds at different time points
    vector_fields = series_pspace.zero()

    # Give the initial two series deformations and series Jacobian determinant
    image_N0 = series_image_space.element()
    grad_data_matching_N1 = series_image_space.element()
    grad_data_matching = image_domain.element(gradS(I))

    if impl=='geom':
        detDphi_N1 = series_image_space.element()
    if impl=='mp':
        detDphi_N0 = series_image_space.element()
        mp_deformed_image_N0 = series_image_space.element()

    for i in range(N+1):
        image_N0[i] = image_domain.element(I).copy()
        if impl=='geom':
            detDphi_N1[i] = image_domain.one()
        if impl=='mp':
            detDphi_N0[i] = image_domain.one()
            mp_deformed_image_N0[i] = image_N0[i].copy()
        grad_data_matching_N1[i] = grad_data_matching.copy()

    # Create the gradient op
    grad_op = Gradient(domain=image_domain, method='forward',
                       pad_mode='symmetric')

    # Create the divergence op
    div_op = Divergence(domain=pspace, method='forward', pad_mode='symmetric')

    # Begin iteration for non-mass-preserving case
    if impl=='geom':
        print(impl)
        for _ in range(niter):
            # Update the velocity field
            for i in range(N+1):
                tmp1 = (grad_data_matching_N1[i] * detDphi_N1[i]).copy()
                tmp = grad_op(image_N0[i]).copy()
                for j in range(dim):
                    tmp[j] *= tmp1
                tmp3 = (2 * np.pi) * vectorial_ft_fit_op.inverse(
                    vectorial_ft_fit_op(tmp) * ft_kernel_fitting)
    
                vector_fields[i] = (vector_fields[i] - eps * (
                    lamb * vector_fields[i] - tmp3)).copy()

            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)
    
            # Update gradient of the data matching: grad S(W_I(v^k))
            grad_data_matching_N1[N] = image_domain.element(
                gradS(PhiStarI)).copy()
            for i in range(N):
                grad_data_matching_N1[N-i-1] = image_domain.element(
                    _linear_deform(grad_data_matching_N1[N-i],
                                   inv_N * vector_fields[N-i-1])).copy()
    
        return image_N0

    # Begin iteration for mass-preserving case
    elif impl=='mp':
        print(impl)
        for _ in range(niter):
            # Update the velocity field
            for i in range(N+1):
                tmp = grad_op(grad_data_matching_N1[i]).copy()
                for j in range(dim):
                    tmp[j] *= mp_deformed_image_N0[i]
                tmp3 = (2 * np.pi) * vectorial_ft_fit_op.inverse(
                    vectorial_ft_fit_op(tmp) * ft_kernel_fitting)
    
                vector_fields[i] = (vector_fields[i] - eps * (
                    lamb * vector_fields[i] + tmp3)).copy()
            
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)
    
            # Update gradient of the data matching: grad S(W_I(v^k))
            grad_data_matching_N1[N] = image_domain.element(
                gradS(PhiStarI)).copy()
            for i in range(N):
                grad_data_matching_N1[N-i-1] = image_domain.element(
                    _linear_deform(grad_data_matching_N1[N-i],
                                   inv_N * vector_fields[N-i-1])).copy()
    
        return mp_deformed_image_N0


# Give input images
#I0name = './pictures/c_highres.png'
#I1name = './pictures/i_highres.png'
#I0name = './pictures/DS0003AxialSlice80.png'
#I1name = './pictures/DS0002AxialSlice80.png'
#I0name = './pictures/hand5.png'
#I1name = './pictures/hand3.png'
I0name = './pictures/handnew1.png'
I1name = './pictures/handnew2.png'
#I0name = './pictures/v.png'
#I1name = './pictures/j.png'
#I0name = './pictures/ImageHalf058.png'
#I1name = './pictures/ImageHalf059.png'
#I0name = './pictures/ImageHalf068.png'
#I1name = './pictures/ImageHalf069.png'

# Get digital images
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = space.element(I0)

# Create the ground truth as the submarine phantom
# ground_truth = submarine_phantom(space, smooth=True, taper=50.0)

# Create the ground truth as the Shepp-Logan phantom
# ground_truth = shepp_logan(space, modified=True)

# Create the template as the given image
template = space.element(I1)

# Create the template as the disc phantom
# template = disc_phantom(space, smooth=True, taper=50.0)

# Create the template as the deformed Shepp-Logan phantom
# template = shepp_logan_2d(space, modified=True)

# Create the template for Shepp-Logan phantom
#deform_field_space = space.vector_field_space
#disp_func = [
#    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
#    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
#deform_field = deform_field_space.element(disp_func)
#template = space.element(geometric_deform(shepp_logan(space, modified=True),
#                                          deform_field))

# FFT setting for data matching term, 1 means 100% padding
padded_size = 2 * space.shape[0]
padded_ft_fit_op = padded_ft_op(space, padded_size)
vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * space.ndim))

# Fix the sigma parameter in the kernel
sigma = 2.5

# Compute the FT of kernel in fitting term
ft_kernel_fitting = fitting_kernel_ft(kernel)

# Maximum iteration number
niter = 1600

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl1 = 'mp'

# Implementation method for image matching or image reconstruction,
# impl chooses 'matching' or 'reconstruction', 'matching' means image matching,
# 'reconstruction' means image reconstruction
impl2 = 'matching'

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if impl1 == 'mp':
#    template *= np.sum(ground_truth) / np.sum(template)
    template *= np.linalg.norm(ground_truth, 'fro')/ \
        np.linalg.norm(template, 'fro')

# Show intermiddle results
callback = CallbackShow(
    '{!r} {!r} iterates'.format(impl1, impl2), display_step=5) & \
    CallbackPrintIteration()

ground_truth.show('ground truth')
template.show('template')

# For image reconstruction
if impl2 == 'reconstruction':
    # Give step size for solver
    eps = 0.005

    # Give regularization parameter
    lamb = 0.0000001

    # Give the number of directions
    num_angles = 10
    
    # Create the uniformly distributed directions
    angle_partition = uniform_partition(0.0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])
    
    # Create 2-D projection domain
    # The length should be 1.5 times of that of the reconstruction space
    detector_partition = uniform_partition(-24, 24, 362)
    
    # Create 2-D parallel projection geometry
    geometry = Parallel2dGeometry(angle_partition, detector_partition)
    
    # Ray transform aka forward projection. We use ASTRA CUDA backend.
    ray_trafo = RayTransform(space, geometry, impl='astra_cuda')

    # Create the forward operator for image reconstruction
    op = ray_trafo

    # Create projection data by calling the op on the phantom
    proj_data = op(ground_truth)

    # Add white Gaussion noise onto the noiseless data
    noise = 0.0 * white_noise(op.range)

    # Add white Gaussion noise from file
    # noise = op.range.element(np.load('noise_20angles.npy'))

    # Create the noisy projection data
    noise_proj_data = proj_data + noise

    # Create the noisy data from file
    #noise_proj_data = op.range.element(
    #    np.load('noise_proj_data_20angles_snr_4_98.npy'))

    # --- Create FilteredBackProjection (FBP) operator --- #    
    # Create FBP operator
    FBP = fbp_op(op, padding=True, filter_type='Hamming',
                 frequency_scaling=0.8)
    # Calculate filtered backprojection of data             
    fbp_reconstruction = FBP(proj_data)
    
    # Shows result of FBP reconstruction
    fbp_reconstruction.show(title='Filtered backprojection')

    # Compute the signal-to-noise ratio in dB
    snr = snr(proj_data, noise, impl='dB')

    # Output the signal-to-noise ratio
    print('snr = {!r}'.format(snr))

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * (op - noise_proj_data)

    # Give the number of time points
    time_itvs = 20

    # Compute by LDDMM solver
    image_N0 = LDDMM_gradient_descent_scheme_solver(
        gradS, template, time_itvs, niter, eps, lamb, impl1, callback)
    
    rec_result_1 = space.element(image_N0[5])
    rec_result_2 = space.element(image_N0[10])
    rec_result_3 = space.element(image_N0[15])
    rec_result = space.element(image_N0[time_itvs])

#    template = rec_result

    # Compute the projections of the reconstructed image
    rec_proj_data = op(rec_result)

    # Plot the results of interest
    plt.figure(1, figsize=(21, 21))
    plt.clf()

    plt.subplot(3, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone',
               vmin=np.asarray(template).min(),
               vmax=np.asarray(template).max())
    plt.colorbar()
    plt.title('Template')
    
    plt.subplot(3, 3, 2)
    plt.imshow(np.rot90(rec_result_1), cmap='bone',
               vmin=np.asarray(rec_result_1).min(),
               vmax=np.asarray(rec_result_1).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(2))

    plt.subplot(3, 3, 3)
    plt.imshow(np.rot90(rec_result_2), cmap='bone',
               vmin=np.asarray(rec_result_2).min(),
               vmax=np.asarray(rec_result_2).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(5))

    plt.subplot(3, 3, 4)
    plt.imshow(np.rot90(rec_result_3), cmap='bone',
               vmin=np.asarray(rec_result_3).min(),
               vmax=np.asarray(rec_result_3).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(8))

    plt.subplot(3, 3, 5)
    plt.imshow(np.rot90(rec_result), cmap='bone',
               vmin=np.asarray(rec_result).min(),
               vmax=np.asarray(rec_result).max()) 
    plt.colorbar()
    plt.title('Reconstructed image by {!r} iters, '
        '{!r} projs'.format(niter, num_angles))

    plt.subplot(3, 3, 6)
    plt.imshow(np.rot90(ground_truth), cmap='bone',
               vmin=np.asarray(ground_truth).min(),
               vmax=np.asarray(ground_truth).max())
    plt.colorbar()
    plt.title('Ground truth')
    
    plt.subplot(3, 3, 7)
    plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(noise_proj_data)[0],
             'r', np.asarray(rec_proj_data)[0], 'g'), 
    plt.axis([0, 181, -3, 10]), plt.grid(True)
#    plt.title('$\Theta=0^\circ$, b: truth, r: noisy, '
#        'g: rec_proj, SNR = {:.3}dB'.format(snr))
#    plt.gca().axes.yaxis.set_ticklabels([])

    plt.subplot(3, 3, 8)
    plt.plot(np.asarray(proj_data)[2], 'b', np.asarray(noise_proj_data)[2],
             'r', np.asarray(rec_proj_data)[2], 'g'),
    plt.axis([0, 181, -3, 10]), plt.grid(True)
#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])

    plt.subplot(3, 3, 9)
    plt.plot(np.asarray(proj_data)[4], 'b', np.asarray(noise_proj_data)[4],
             'r', np.asarray(rec_proj_data)[4], 'g'),
    plt.axis([0, 181, -3, 10]), plt.grid(True)
#    plt.title('$\Theta=162^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])

# For image matching
if impl2 == 'matching':
    # Give step size for solver
    eps = 0.05

    # Give regularization parameter
    lamb = 0.0000001

    # Create the forward operator for image matching
    op = IdentityOperator(space)

    # Create data by calling the op on the phantom
    data = op(ground_truth)

    # Add white Gaussion noise onto the noiseless data
    noise = 0.0 * white_noise(op.range)

    # Create the noisy projection data
    noise_data = data + noise

    # Compute the signal-to-noise ratio in dB
    snr = snr(data, noise, impl='dB')

    # Output the signal-to-noise ratio
    print('snr = {!r}'.format(snr))

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * (op - noise_data)

    # Give the number of time intervals
    time_itvs = 10

    # Compute by LDDMM solver
    image_N0 = LDDMM_gradient_descent_scheme_solver(
        gradS, template, time_itvs, niter, eps, lamb, impl1, callback)
    
    rec_result_1 = space.element(image_N0[2])
    rec_result_2 = space.element(image_N0[5])
    rec_result_3 = space.element(image_N0[8])
    rec_result = space.element(image_N0[time_itvs])

    # Plot the results of interest
    plt.figure(1, figsize=(21, 10))
    plt.clf()

    plt.subplot(2, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone',
               vmin=np.asarray(template).min(),
               vmax=np.asarray(template).max())
    plt.colorbar()
    plt.title('Template')
    
    plt.subplot(2, 3, 2)
    plt.imshow(np.rot90(rec_result_1), cmap='bone',
               vmin=np.asarray(rec_result_1).min(),
               vmax=np.asarray(rec_result_1).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(2))

    plt.subplot(2, 3, 3)
    plt.imshow(np.rot90(rec_result_2), cmap='bone',
               vmin=np.asarray(rec_result_2).min(),
               vmax=np.asarray(rec_result_2).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(5))

    plt.subplot(2, 3, 4)
    plt.imshow(np.rot90(rec_result_3), cmap='bone',
               vmin=np.asarray(rec_result_3).min(),
               vmax=np.asarray(rec_result_3).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(8))

    plt.subplot(2, 3, 5)
    plt.imshow(np.rot90(rec_result), cmap='bone',
               vmin=np.asarray(rec_result).min(),
               vmax=np.asarray(rec_result).max()) 
    plt.colorbar()
    plt.title('Imatched image by {!r} iters'.format(niter))

    plt.subplot(2, 3, 6)
    plt.imshow(np.rot90(ground_truth), cmap='bone',
               vmin=np.asarray(ground_truth).min(),
               vmax=np.asarray(ground_truth).max())
    plt.colorbar()
    plt.title('Ground truth')
