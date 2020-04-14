import math
import numba.cuda as cuda

class ProjectConfig (object):
	def __init__(self, shape:tuple, d=3, block=(16, 16, 3)):
		self.height = shape[1]  # heightß
		self.width = shape[0]  # width
		# assert (isinstance(self.height,int) and isinstance(self.width, int))
		# assert (2 * self.height == self.width and self.height % 16 == 0)

		self.depth = d
		self.edge= int(self.width / 4) # edge
		self.block_dim = block
		self.grid_dim = (int (self.edge / self.block_dim[0]), int (self.edge / self.block_dim[1]), 1)
		self.view_shape = [self.edge, self.edge, self.depth]

@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_front(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = 1.0
	y: float = (2 * i / edge) - 1
	z: float = 1 - (2 * j / edge)

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]


@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_right(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = 1 - (2 * i / edge)
	y: float = 1.0
	z: float = 1 - (2 * j / edge)

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]


@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_back(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = - 1.0
	y: float = 1 - (2 * i / edge)
	z: float = 1 - (2 * j / edge)

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]

@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_left(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = (2 * i / edge) - 1
	y: float = -1.0
	z: float = 1 - (2 * j / edge)

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]


@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_up(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = (2 * j / edge) - 1
	y: float = 1 - (2 * i / edge)
	z: float = 1.0

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]


@cuda.jit ('void(uint8[:,:,:],uint8[:,:,:])')
def project_down(dest, src):
	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	k = cuda.threadIdx.z
	edge: int = cuda.blockDim.x * cuda.gridDim.x
	edge2 = edge * 2
	edge4 = edge * 4

	x: float = 1 - (2 * j / edge)
	y: float = 1 - (2 * i / edge)
	z: float = - 1.0

	theta = math.atan2 (y, x)
	phi = math.atan2 (z, math.hypot (x, y))

	uf = edge2 * (theta + math.pi) / math.pi
	vf = edge2 * (math.pi / 2 - phi) / math.pi

	u1: int = int (math.floor (uf))
	v1: int = int (math.floor (vf))

	u1 = u1 % edge4
	if v1 > edge2 - 1:
		v1 = edge2 - 1

	dest[j, i, k] = src[v1, u1, k]
