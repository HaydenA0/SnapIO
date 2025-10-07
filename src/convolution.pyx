import numpy as np
from cython.parallel import prange
cimport numpy as np
import cython

@cython.boundscheck(False) 
@cython.wraparound(False)  
def apply_kernel_cython(double[:,:] image, double[:,:] rotated_kernel ):
    cdef int i, j, ky,kx
    cdef int img_height = image.shape[0]
    cdef int img_width = image.shape[1]
    cdef int kernel_height = rotated_kernel.shape[0]
    cdef int kernel_width = rotated_kernel.shape[1]
    cdef int floor_kernel_height = kernel_height // 2
    cdef int floor_kernel_width = kernel_width // 2
    output_np = np.zeros_like(image)
    cdef double[:, :] output_view = output_np
    cdef double accumulator 
    cdef int img_y_base
    cdef int img_x_base 

    cdef int starting_point_i = floor_kernel_height
    cdef int starting_point_j = floor_kernel_width
    cdef int finishing_point_i = img_height - (kernel_height - floor_kernel_height) + 1
    cdef int finishing_point_j = img_width - (kernel_width - floor_kernel_width) + 1
    with nogil :
        for i in prange(starting_point_i, finishing_point_i):
            for j in range(starting_point_j, finishing_point_j):
                    accumulator = 0.0
                    img_y_base = i - kernel_height//2 
                    img_x_base = j - kernel_width//2 
                    for ky in range(kernel_height):
                        for kx in range(kernel_width):
                            accumulator = accumulator + image[img_y_base + ky, img_x_base+kx] * rotated_kernel[ky, kx]
                            output_view[i,j] = accumulator
    return output_np
