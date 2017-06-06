import numpy as np
import numpy.linalg
import math
import time
from utils.io import *
from node import *
from scipy.spatial.distance import euclidean
from scipy.fftpack import fftn, ifftn, ifft
from scipy.special import jv


def main():
    img = loadimg('test/Frog/crop_region.tif')
    # print(img.shape,np.max(img),np.min(img))
    # img *= 7
    # writetiff3d('test/Frog/enhanced_region.tif',img)

    filtered_region = response(img.astype('float'),np.asarray(np.arange(10, 15, 1)))[0]
    # filtered_region[np.argwhere(filtered_region>0)] = 255
    writetiff3d('test/Frog/test_region.tif',filtered_region)
    # anisotropicfilter(img)
    # print(img1.shape,np.max(img1),np.min(img1))

def anisotropicfilter(img):
    x = np.asarray(hessian3(img))
    print(x.shape)
    np.linalg.eig(x)

def hessian3(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    tmpgrad = np.gradient(x_grad[0])
    f11 = tmpgrad[0]
    f12 = tmpgrad[1]
    f13 = tmpgrad[2]
    tmpgrad = np.gradient(x_grad[1])
    f22 = tmpgrad[1]
    f23 = tmpgrad[2]
    tmpgrad = np.gradient(x_grad[2])
    f33 = tmpgrad[2]
    return [f11, f12, f13, f22, f23, f33]

def dist_gradient(self):
        t = skfmm.travel_time(marchmap, speed, dx=5e-3)
        fx = np.zeros(shape=t.shape)
        fy = np.zeros(shape=t.shape)
        fz = np.zeros(shape=t.shape)

        J = np.zeros(shape=[s + 2 for s in t.shape])  # Padded Image
        J[:, :, :] = t.max()
        J[1:-1, 1:-1, 1:-1] = t
        Ne = [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0],
              [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1], [0, -1, -1],
              [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 1], [0, 1, -1],
              [0, 1, 0], [0, 1, 1], [1, -1, -1], [1, -1, 0], [1, -1, 1],
              [1, 0, -1], [1, 0, 0], [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]]

        for n in Ne:
            In = J[1 + n[0]:J.shape[0] - 1 + n[0], 1 + n[1]:J.shape[1] - 1 + n[1],
                   1 + n[2]:J.shape[2] - 1 + n[2]]
            check = In < t
            t[check] = In[check]
            D = np.divide(n, np.linalg.norm(n))
            fx[check] = D[0]
            fy[check] = D[1]
            fz[check] = D[2]
        return -fx, -fy, -fz


def response(img, radii,rsptype='oof'):
    eps = 1e-12
    rsp = np.zeros(img.shape)
    # bar = progressbar.ProgressBar(max_value=kwargs['radii'].size)
    # bar.update(0)

    W = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3)) # Eigen values to save
    V = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3, 3)) # Eigen vectors to save

    if rsptype == 'oof' :
        rsptensor = ooftensor(img, radii)

    # pbar = tqdm(total=len(radii))
    for i, tensorfield in enumerate(rsptensor):
        # Make the tensor from tensorfield
        f11, f12, f13, f22, f23, f33 = tensorfield
        tensor = np.stack((f11, f12, f13, f12, f22, f23, f13, f23, f33), axis=-1)
        del f11
        del f12
        del f13
        del f22
        del f23
        del f33
        tensor = tensor.reshape(img.shape[0], img.shape[1], img.shape[2], 3, 3)
        w, v = np.linalg.eigh(tensor)
        del tensor
        sume = w.sum(axis=-1)
        nvox = img.shape[0] * img.shape[1] * img.shape[2]
        sortidx = np.argsort(np.abs(w), axis=-1)
        sortidx = sortidx.reshape((nvox, 3))

        # Sort eigenvalues according to their abs
        w = w.reshape((nvox, 3))
        for j, (idx, value) in enumerate(zip(sortidx, w)):
            w[j,:] = value[idx]
        w = w.reshape(img.shape[0], img.shape[1], img.shape[2], 3)

        # Sort eigenvectors according to their abs
        v = v.reshape((nvox, 3, 3))
        for j, (idx, vec) in enumerate(zip(sortidx, v)):
            v[j,:,:] = vec[:, idx]
        del sortidx
        v = v.reshape(img.shape[0], img.shape[1], img.shape[2], 3, 3)

        mine = w[:,:,:, 0]
        mide = w[:,:,:, 1]
        maxe = w[:,:,:, 2]

        if rsptype == 'oof':
            feat = maxe
        elif rsptype == 'bg':
            feat = -mide / maxe * (mide + maxe) # Medialness measure response
            cond = sume >= 0
            feat[cond] = 0 # Filter the non-anisotropic voxels

        del mine
        del maxe
        del mide
        del sume

        cond = np.abs(feat) > np.abs(rsp)
        W[cond, :] = w[cond, :]
        V[cond, :, :] = v[cond, :, :]
        rsp[cond] = feat[cond]
        del v
        del w
        del tensorfield
        del feat
        del cond
        # pbar.update(1)

    return rsp, V, W



def ooftensor(img, radii, memory_save=True):
    '''
    type: oof, bg
    '''
    # sigma = 1 # TODO: Pixel spacing
    eps = 1e-12
    # ntype = 1 # The type of normalisation
    fimg = fftn(img, overwrite_x=True)
    shiftmat = ifftshiftedcoormatrix(fimg.shape)
    x, y, z = shiftmat
    x = x / fimg.shape[0]
    y = y / fimg.shape[1]
    z = z / fimg.shape[2]
    kernel_radius = np.sqrt(x ** 2 + y ** 2 + z ** 2) + eps # The distance from origin

    for r in radii:
        # Make the fourier convolutional kernel
        jvbuffer = oofftkernel(kernel_radius, r) * fimg

        if memory_save:
            # F11
            buffer = ifftshiftedcoordinate(img.shape, 0) ** 2 * x * x * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f11 = buffer.copy()

            # F12
            buffer = ifftshiftedcoordinate(img.shape, 0) * ifftshiftedcoordinate(img.shape, 1) * x * y * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f12 = buffer.copy()

            # F13
            buffer = ifftshiftedcoordinate(img.shape, 0) * ifftshiftedcoordinate(img.shape, 2) * x * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f13 = buffer.copy()

            # F22
            buffer = ifftshiftedcoordinate(img.shape, 1) ** 2 * y ** 2 * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f22 = buffer.copy()

            # F23
            buffer = ifftshiftedcoordinate(img.shape, 1) * ifftshiftedcoordinate(img.shape, 2) * y * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f23 = buffer.copy()

            # F33
            buffer = ifftshiftedcoordinate(img.shape, 2) * ifftshiftedcoordinate(img.shape, 2) * z * z * jvbuffer
            buffer = ifft(buffer, axis=0)
            buffer = ifft(buffer, axis=1)
            buffer = ifft(buffer, axis=2)
            f33 = buffer.copy()
        else:
            f11 = np.real(ifftn(x * x * jvbuffer))
            f12 = np.real(ifftn(x * y * jvbuffer))
            f13 = np.real(ifftn(x * z * jvbuffer))
            f22 = np.real(ifftn(y * y * jvbuffer))
            f23 = np.real(ifftn(y * z * jvbuffer))
            f33 = np.real(ifftn(z * z * jvbuffer))
        yield [f11, f12, f13, f22, f23, f33]

def ifftshiftedcoormatrix(shape):
    shape = np.asarray(shape)
    p = np.floor(np.asarray(shape) / 2).astype('int')
    coord = []
    for i in range(shape.size):
        a = np.hstack((np.arange(p[i], shape[i]), np.arange(0, p[i]))) - p[i] - 1.
        repmatpara = np.ones((shape.size,)).astype('int')
        repmatpara[i] = shape[i]
        A = a.reshape(repmatpara)
        repmatpara = shape.copy()
        repmatpara[i] = 1
        coord.append(np.tile(A, repmatpara))

    return coord


def ifftshiftedcoordinate(shape, axis):
    shape = np.asarray(shape)
    p = np.floor(np.asarray(shape) / 2).astype('int')
    a = (np.hstack((np.arange(p[axis], shape[axis]), np.arange(0, p[axis]))) - p[axis] - 1.).astype('float')
    a /= shape[axis].astype('float')
    reshapepara = np.ones((shape.size,)).astype('int');
    reshapepara[axis] = shape[axis];
    A = a.reshape(reshapepara);
    repmatpara = shape.copy();
    repmatpara[axis] = 1;
    return np.tile(A, repmatpara)

def oofftkernel(kernel_radius, r, sigma=1, ntype=1):
    eps = 1e-12
    normalisation = 4/3 * np.pi * r**3 / (jv(1.5, 2*np.pi*r*eps) / eps ** (3/2)) / r**2 *  \
                    (r / np.sqrt(2.*r*sigma - sigma**2)) ** ntype
    jvbuffer = normalisation * np.exp( (-2 * sigma**2 * np.pi**2 * kernel_radius**2) / (kernel_radius**(3/2) ))
    return (np.sin(2 * np.pi * r * kernel_radius) / (2 * np.pi * r * kernel_radius) - np.cos(2 * np.pi * r * kernel_radius)) * \
               jvbuffer * np.sqrt( 1./ (np.pi**2 * r *kernel_radius ))

if __name__ == "__main__":
    main()