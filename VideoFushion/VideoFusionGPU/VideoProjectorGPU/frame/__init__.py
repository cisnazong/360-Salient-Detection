import numpy as np
import numba.cuda as cuda
from ..utils.ImageProjectorCubicBilinear import project_front, project_right, project_back, project_left, project_up, \
    project_down


class frame_processor(object):
    def __init__(self, frame_shape, view_shape=(512, 512, 3), block=(16, 16, 1)):
        self.valid = False
        self.view_shape = view_shape
        self.block_dim = block
        assert block[0] == block[1]
        assert view_shape[0] == view_shape[1]
        assert view_shape[0] % block[0] == 0

        self.frame_shape = frame_shape
        self.height = frame_shape[0]
        self.width = frame_shape[1]
        self.depth = frame_shape[2]

        self.valid = True
        self.edge = view_shape[0]
        self.grid_dim = (int(self.view_shape[0] / self.block_dim[0]), int(self.view_shape[1] / self.block_dim[1]), 1)

        self.function_list = [project_front, project_right, project_back, project_left, project_up, project_down]
        self.stream_list = [cuda.stream() for i in range(6)]
        self.stream0 = cuda.stream()
        self.stream1 = cuda.stream()

        self.imgIn_gpu = cuda.device_array(shape=self.frame_shape, dtype=np.uint8, stream=self.stream0)
        self.imgOut_gpu_list = [cuda.device_array(shape=self.view_shape,
                                                  dtype=np.uint8,
                                                  stream=self.stream_list[i]) for i in range(6)]

        self.imgOut_cpu_list = [np.zeros(shape=self.view_shape, dtype=np.uint8) for i in range(6)]
        self.title_list = ['front', 'right', 'back', 'left', 'up', 'down']

    def render(self, frame):
        pass

class frame_to_list(frame_processor):
    def __init__(self, frame_shape, view_shape=(512, 512, 3), block=(16, 16, 1)):
        super(frame_to_list, self).__init__(frame_shape, view_shape, block)

    def render(self, frame: np.array):
        width = frame.shape[1]
        height = frame.shape[0]
        self.imgIn_gpu = cuda.to_device(frame, stream=self.stream0)

        for i in range(6):
            self.function_list[i][self.grid_dim, self.block_dim, self.stream_list[i]] \
                (self.imgOut_gpu_list[i], self.imgIn_gpu, height, width, self.edge)
            self.imgOut_gpu_list[i].copy_to_host(self.imgOut_cpu_list[i], stream=self.stream_list[i])
        cuda.synchronize()
        return self.imgOut_cpu_list


class frame_to_horizon(frame_processor):
    def __init__(self, frame_shape, view_shape=(512, 512, 3), block=(16, 16, 1)):
        super(frame_to_horizon, self).__init__(frame_shape, view_shape, block)
        self.imgOut_cpu = np.zeros(shape=(self.edge * 6, self.edge, self.depth), dtype=np.uint8)

    def render(self, frame: np.array):
        width = frame.shape[1]
        height = frame.shape[0]
        self.imgIn_gpu = cuda.to_device(frame, stream=self.stream0)

        for i in range(6):
            self.function_list[i][self.grid_dim, self.block_dim, self.stream_list[i]] \
                (self.imgOut_gpu_list[i], self.imgIn_gpu, height, width, self.edge)
            self.imgOut_gpu_list[i].copy_to_host(self.imgOut_cpu_list[i], stream=self.stream_list[i])
        cuda.synchronize()
        self.imgOut_cpu = np.concatenate(tuple([self.imgOut_cpu_list[i] for i in range(6)]), axis=1)
        return self.imgOut_cpu
