import numpy as np


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
