# ./visulization.py
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self) -> None:
        pass

    def compare_images(self, images_list, titles):
        images_list = [np.array(image) for image in images_list]
        if len(images_list) != len(titles):
            raise ValueError("Length of images_list and titles must be the same")

        n = len(images_list)
        plt.figure(figsize=(4 * n, 4))

        for i, (img, title) in enumerate(zip(images_list, titles)):
            plt.subplot(1, n, i + 1)
            if img.ndim == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def show_histogram(self, images_list):
        images_list = [np.array(image) for image in images_list]

        n = len(images_list)
        plt.figure(figsize=(4 * n, 4))

        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.hist(images_list[i])

        plt.tight_layout()
        plt.show()
