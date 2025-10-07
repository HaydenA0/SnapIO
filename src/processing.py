# ./processing.py
import numpy as np
import convolution


def rotate_kernel(kernel):
    return np.rot90(kernel, 2)


def convolution_kernel_segment(kernel, patch):
    return np.sum(kernel * patch)


class Processor:
    def __init__(self) -> None:
        pass

    def calculate_normal_difference_of_images(self, img1, img2):
        img1_bigger = img1.astype(np.float32) / 255
        img2_bigger = img2.astype(np.float32) / 255
        x1, y1 = img1.shape[:2]
        x2, y2 = img2.shape[:2]
        if x1 != x2 or y1 != y2:
            raise ValueError(
                f"The Image 1 has a size of {(x1,y1)} and the Image 2 has a size of {(x2,y2)}. \n Make sure they are the smae dimension. "
            )
        return np.sum((img1_bigger - img2_bigger) ** 2) / (x1 * y1)

    def apply_kernel_old(self, image, kernel):
        n, m = kernel.shape
        rotated_kernel = rotate_kernel(kernel)
        output = np.zeros_like(image)
        floor_n = n // 2
        floor_m = m // 2
        ceil_n = n - floor_n
        ceil_m = m - floor_m
        for i in range(floor_n, image.shape[0] - ceil_n + 1):
            for j in range(floor_m, image.shape[1] - ceil_m + 1):
                patch = image[i - floor_n : i + ceil_n, j - floor_m : j + ceil_m]
                output[i, j] = convolution_kernel_segment(rotated_kernel, patch)
        return output

    def apply_kernel_new(self, image, kernel):
        rotated_kernel = rotate_kernel(kernel)
        output_image = convolution.apply_kernel_cython(image, rotated_kernel)
        return output_image

    def median_filter(self, image, grid_size: int = 3):
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array.")

        if image.ndim != 2:
            raise ValueError("Image must be 2D .")

        result = image.copy()
        floor_n = grid_size // 2
        ceil_n = grid_size - floor_n
        for i in range(floor_n, image.shape[0] - ceil_n + 1):
            for j in range(floor_n, image.shape[1] - ceil_n + 1):
                patch = image[i - floor_n : i + ceil_n, j - floor_n : j + ceil_n]
                patch.sort()
                median = patch[len(patch) // 2]
                result[i, j] = median

        return result
