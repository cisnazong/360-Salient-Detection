import numpy as np
import numba.cuda as cuda
import math
import numba


@cuda.jit('void(float32,float32,uint32, float32[:])', device=True)
def sphere2cube(theta, phi, edge, coord):
    x = 0.5 * math.sin(theta) * math.cos(phi)
    y = 0.5 * math.sin(theta) * math.sin(phi)
    z = 0.5 * math.cos(theta)
    scale = 2 * max(abs(x), abs(y), abs(z))
    x, y, z = x / scale, y / scale, z / scale
    if abs(x + 0.5) < 0.000001:
        coord[1] = ((2.5 + y) * edge) % (4 * edge)
        coord[0] = (0.5 - z) * edge
        return

    if abs(x - 0.5) < 0.000001:
        coord[1] = ((0.5 - y) * edge) % (4 * edge)
        coord[0]  = (0.5 - z) * edge
        return

    if abs(y - 0.5) < 0.000001:
        coord[1] = ((3.5 + x) * edge )% (4 * edge)
        coord[0]  = (0.5 - z) * edge
        return

    if abs(y + 0.5) < 0.000001:
        coord[1] = ((1.5 - x) * edge) % (4 * edge)
        coord[0]  = (0.5 - z) * edge
        return

    if abs(z - 0.5) < 0.000001:
        coord[1] = (4.5 - y) * edge
        coord[0]  = (0.5 + x) * edge
        return

    if abs(z + 0.5) < 0.000001:
        coord[1] = (5.5 - y) * edge
        coord[0] = (0.5 - x) * edge
        return


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:],int32,int32,int32)')
def project(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    phi = (0.5 - j / w) * 2 * math.pi
    theta = (i / h) * math.pi
    coord = cuda.local.array(shape=2,dtype=numba.float32)
    sphere2cube(theta, phi, edge, coord)

    uf, vf = coord[1], coord[0]

    u1: int = int(math.floor (uf))
    v1: int = int(math.floor (vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    if v2 >= edge: v2 = edge - 1
    if u2 / edge != u1 /edge: u2 = u1

    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[i, j, k] = int (math.floor (p0 + p1 + p2 + p3))

@cuda.jit('void(uint8[:,:,:],uint8[:,:,:])')
def copy(dest, src):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    for k in range(3):
        dest[i, j, k] = src[i, j, k]
