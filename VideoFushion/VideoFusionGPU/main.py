import numba
import numpy as np
import numba.cuda as cuda

from utils import sphere2cube
if __name__ == '__main__':
    result = np.array([0,0], np.float32)
    # sphere2cube(0, 0, 5000, result)
    # sphere2cube(0, 0, 5000, result)
    print(result)