import pycuda.autoinit  # very important import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from VideoProjectorCUDA.utils.ImageProjectCUDA import ImageProjectorTarget
import multiprocessing
import numpy as np
import threading


def project_front(target, imgIn):
	cuda.init()
	mod = SourceModule (
		"""
		#include <stdint.h>
		__global__ void project_front(uint8_t * dest, uint8_t * source, int * e)
            {
              int edge = e[0];
              int edge4 = edge * 4;
              int edge2 = edge * 2;
              int i = (blockIdx.x * blockDim.x + threadIdx.x);
              int j = (blockIdx.y * blockDim.y + threadIdx.y);
              int k = threadIdx.z;
              
              float x = 1.0;
              float y = 2 * fdivide(i, edge) - 1;
              float z = 1 - 2 * fdivide(j, edge);
              
              float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge) + i / 4) * 3 + k] = source[(v2 * edge4 + u2) * 3 + k];
              dest[((j * edge) + i) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
            }
		"""
	)

	f = mod.get_function ("project_front")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)


def project_right(target, imgIn):
	mod = SourceModule (
		"""
		#include <stdint.h>
		__global__ void project_right(uint8_t * dest, uint8_t * source, int * e)
			{
			  int edge = e[0];
			  int edge4 = edge * 4;
			  int edge2 = edge * 2;
			  int i = (blockIdx.x * blockDim.x + threadIdx.x);
			  int j = (blockIdx.y * blockDim.y + threadIdx.y);
			  int k = threadIdx.z;

			  double x = 1 - 2 * fdivide(i, edge);
			  double y = 1.0;
			  double z = 1 - 2 * fdivide(j, edge);

			  float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge / 4) + i / 4) * 3 + k] = source[(v2 * edge + u2 / 4) * 3 + k];
              dest[((j * edge) + i ) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
			}
		"""
	)
	f = mod.get_function ("project_right")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)


def project_back(target, imgIn):
	mod = SourceModule (
		"""
		#include <stdint.h>
		__global__ void project_back(uint8_t * dest, uint8_t * source, int * e)
            {
              int edge = e[0];
              int edge4 = edge * 4;
              int edge2 = edge * 2;
              int i = (blockIdx.x * blockDim.x + threadIdx.x);
              int j = (blockIdx.y * blockDim.y + threadIdx.y);
              int k = threadIdx.z;
              
              double x = - 1.0;
              double y = 1 - 2 * fdivide(i, edge);
              double z = 1 - 2 * fdivide(j, edge);
              
                            float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge / 4) + i / 4) * 3 + k] = source[(v2 * edge + u2 / 4) * 3 + k];
              dest[((j * edge) + i ) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
            }
		"""
	)
	f = mod.get_function ("project_back")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)


def project_left(target, imgIn):
	mod = SourceModule (
		"""
		#include <stdint.h>
		 __global__ void project_left(uint8_t * dest, uint8_t * source, int * e)
            {
              int edge = e[0];
              int edge4 = edge * 4;
              int edge2 = edge * 2;
              int i = (blockIdx.x * blockDim.x + threadIdx.x);
              int j = (blockIdx.y * blockDim.y + threadIdx.y);
              int k = threadIdx.z;
              
              double x = 2 * fdivide(i, edge) - 1;
              double y = - 1.0;
              double z = 1 - 2 * fdivide(j, edge);
              
              float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge / 4) + i / 4) * 3 + k] = source[(v2 * edge + u2 / 4) * 3 + k];
              dest[((j * edge) + i ) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
            }
		"""
	)
	f = mod.get_function ("project_left")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)


def project_up(target, imgIn):
	mod = SourceModule (
		"""
		#include <stdint.h>
		__global__ void project_up(uint8_t * dest, uint8_t * source, int * e)
            {
              int edge = e[0];
              int edge4 = edge * 4;
              int edge2 = edge * 2;
              int i = (blockIdx.x * blockDim.x + threadIdx.x);
              int j = (blockIdx.y * blockDim.y + threadIdx.y);
              int k = threadIdx.z;
              
              double x = 2 * fdivide(j, edge) - 1;
              double y = 1 - 2 * fdivide(i, edge);
              double z = 1.0;
              
              float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge / 4) + i / 4) * 3 + k] = source[(v2 * edge + u2 / 4) * 3 + k];
              dest[((j * edge) + i ) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
            }
		"""
	)
	f = mod.get_function ("project_up")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)


def project_down(target, imgIn):
	mod = SourceModule (
		"""
		#include <stdint.h>
		__global__ void project_down(uint8_t * dest, uint8_t * source, int * e)
            {
              int edge = e[0];
              int edge4 = edge * 4;
              int edge2 = edge * 2;
              int i = (blockIdx.x * blockDim.x + threadIdx.x);
              int j = (blockIdx.y * blockDim.y + threadIdx.y);
              int k = threadIdx.z;
              
              float x = 1 - 2 * fdivide(j, edge);
              float y = 1 - 2 * fdivide(i, edge);
              float z = - 1.0;
              
                            float theta = atan2(y, x);
              float phi = atan2(z, hypot(x, y));
              
              float uf = fdivide(edge2 * (theta + 3.14159265358979), 3.14159265358979);
              float vf = fdivide(edge2 * (1.570796326794895 - phi), 3.14159265358979);
              
              int u1 = __float2uint_rn(uf);
              int v1 = __float2uint_rn(vf);
              double  mu = uf - u1;
              double  mv = vf - v1;
              int u2 = u1 + 1;
              int v2 = v1 + 1;
              
              u1 = u1 % edge4;
              u2 = u2 % edge4;
              if (v1 <= 0) v1 =0;
              if (v1 > edge2 - 1) v1 = edge2 - 1;
              if (v2 <= 0) v2 =0;
              if (v2 > edge2 - 1) v1 = edge2 - 1;
              
              float p0 =  __uint2float_rn(source[(v1 * edge4 + u1) * 3 + k]) * (1 - mu) * (1 - mv);
              float p1 =  __uint2float_rn(source[(v1 * edge4 + u2) * 3 + k]) * mu * (1 - mv) ;
              float p2 =  __uint2float_rn(source[(v2 * edge4 + u1) * 3 + k]) * mv * (1 - mu);
              float p3 =  __uint2float_rn(source[(v2 * edge4 + u2) * 3 + k]) * mv * mu;
              			
              //dest[((j * edge / 4) + i / 4) * 3 + k] = source[(v2 * edge + u2 / 4) * 3 + k];
              dest[((j * edge) + i ) * 3 + k] = __float2uint_rn(p3 + p1 +p2 + p0);
            }
		"""
	)
	f = mod.get_function ("project_down")
	f (cuda.Out (target.Output), cuda.In (imgIn), cuda.In (target.edge), block=target.block_dim,
	   grid=target.grid_dim)

def project_all(target, imgIn):
	pass

def project(T: ImageProjectorTarget, imgIn):
	project_front (T, imgIn)
	# project_left (T, imgIn)
	# project_back (T, imgIn)
	# project_right (T, imgIn)
	# project_up (T, imgIn)
	# project_down (T, imgIn)

	return T.Output
