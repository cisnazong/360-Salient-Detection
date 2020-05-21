import numpy as np
import numba.cuda as cuda
from utils import sphere2cube, project


class frame_processor(object):
    def __init__(self, frame_shape, view_shape=(1024, 2048, 3), block=(16, 16, 1)):
        self.valid = False
        self.view_shape = view_shape
        self.block_dim = block
        assert block[0] == block[1]
        assert view_shape[0] * 2 == view_shape[1]
        assert view_shape[0] % block[0] == 0
        assert frame_shape[1] == 6 * frame_shape[0]

        self.valid = True
        self.frame_shape = frame_shape
        self.height = view_shape[0]
        self.width = view_shape[1]
        self.depth = view_shape[2]
        self.edge = frame_shape[0]
        self.grid_dim = (int(self.view_shape[0] / self.block_dim[0]), int(self.view_shape[1] / self.block_dim[1]), 1)

        self.function = project

        self.stream0 = cuda.stream()
        self.stream1 = cuda.stream()

        self.imgIn_gpu = cuda.device_array(shape=self.frame_shape, dtype=np.uint8, stream=self.stream0)
        self.imgOut_gpu = cuda.device_array(shape=self.view_shape, dtype=np.uint8, stream=self.stream1)
        self.imgOut_cpu = np.zeros(shape=self.view_shape, dtype=np.uint8)

    def render(self, frame):
        pass


class frame_from_horizon(frame_processor):
    def __init__(self, frame_shape, view_shape=(1024, 2048, 3), block=(16, 16, 1)):
        super(frame_from_horizon, self).__init__(frame_shape, view_shape, block)

    def render(self, frame: np.array):
        cuda.from_cuda_array_interface
        self.imgIn_gpu = cuda.to_device(frame)
        self.function[self.grid_dim, self.block_dim](self.imgOut_gpu, self.imgIn_gpu, self.height, self.width, self.edge)
        self.imgOut_gpu.copy_to_host(self.imgOut_cpu)
        cuda.synchronize()
        return self.imgOut_cpu
