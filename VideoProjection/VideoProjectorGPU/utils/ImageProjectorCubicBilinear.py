import math
import numba.cuda as cuda


# This is a class that stores configuration for the projection


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_front(dest, src, h, w, edge):
    # index pixels
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # convert pixel position to geological position
    x: float = 1.0
    y: float = (2 * i / edge) - 1
    z: float = 1 - (2 * j / edge)

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_right(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    x: float = 1 - (2 * i / edge)
    y: float = 1.0
    z: float = 1 - (2 * j / edge)

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_back(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    x: float = - 1.0
    y: float = 1 - (2 * i / edge)
    z: float = 1 - (2 * j / edge)

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_left(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    x: float = (2 * i / edge) - 1
    y: float = -1.0
    z: float = 1 - (2 * j / edge)

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_up(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    x: float = (2 * j / edge) - 1
    y: float = (2 * i / edge) - 1
    z: float = 1.0

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))


@cuda.jit('void(uint8[:,:,:],uint8[:,:,:], int32, int32, int32)')
def project_down(dest, src, h, w, edge):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    x: float = 1 - (2 * j / edge)
    y: float = (2 * i / edge) - 1
    z: float = - 1.0

    # convert (x,y,z) to (r, phi, theta)
    phi = math.atan2(y, x)
    theta = 0.5 * math.pi - math.atan2(z, math.hypot(x, y))

    # map to (u,v) coordinates
    uf = (w * (phi + math.pi) / (2 * math.pi))
    vf = h * (theta / math.pi)

    u1: int = int(math.floor(uf))
    v1: int = int(math.floor(vf))
    mu: float = uf - u1
    mv: float = vf - v1
    u2: int = u1 + 1
    v2: int = v1 + 1

    u1 = u1 % w
    u2 = u2 % w

    if v1 < 0: v1 = 0
    if v2 < 0: v2 = 0

    if v1 > h - 1: v1 = h - 1
    if v2 > h - 1: v2 = h - 1

    # bi-linear combination
    for k in range(3):
        p0: float = src[v1, u1, k] * (1 - mu) * (1 - mv)
        p1: float = src[v1, u2, k] * mu * (1 - mv)
        p2: float = src[v2, u1, k] * (1 - mu) * mv
        p3: float = src[v2, u2, k] * mu * mv
        dest[j, i, k] = int(math.floor(p0 + p1 + p2 + p3))
