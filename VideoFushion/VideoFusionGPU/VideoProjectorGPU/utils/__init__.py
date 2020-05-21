class ProjectConfig(object):
    def __init__(self, shape: tuple, view_shape=(512, 512, 3), d=3, block=(16, 16, 1)):
        self.height = shape[1]  # heightß
        self.width = shape[0]  # width
        self.view_shape = view_shape

        # Uncomment to align blocks and grids
        # assert (isinstance(self.height,int) and isinstance(self.width, int))
        # assert (2 * self.height == self.width and self.height % 16 == 0)

        self.depth = d  # number of color channels
        self.edge = int(self.width / 4)  # length of the edge of each error

        # CUDA configuration
        self.block_dim = block
        self.grid_dim = (int(self.view_shape[0] / self.block_dim[0]) , int(self.view_shape[1] / self.block_dim[1]), 1)

