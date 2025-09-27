from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


class ImageIO:
    def __init__(self) -> None:
        pass

    def load_image(self, image_path, format="RGB", numerical: bool = True):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            img = Image.open(image_path)
        except Exception as e:
            raise IOError(f"Error opening image: {e}")

        valid_formats = ["RGB", "RGBA", "L", "Grayscale"]
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format: {format}. Choose from {valid_formats}"
            )

        if format == "Grayscale":
            img = img.convert("L")
        else:
            img = img.convert(format)

        if numerical:
            return np.array(img)
        return img

    def save_image(
        self, img, image_name: str, is_numerical: bool = True, image_path="images/"
    ):

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        if is_numerical:
            try:
                img = Image.fromarray(img)
            except Exception as e:
                raise ValueError(f"Cannot convert numerical array to image: {e}")

        try:
            img.save(f"{image_path}{image_name}")
        except Exception as e:
            raise IOError(f"Error saving image: {e}")

    def inspect_image(self, img, is_numerical: bool = True):
        if is_numerical:
            if not isinstance(img, np.ndarray):
                raise TypeError("Expected a numpy array for numerical inspection")
            print(f"Image demensions = {img.shape}")
            if img.ndim == 2:
                print("Mode: Grayscale")
            elif img.ndim == 3:
                if img.shape[2] == 3:
                    print("Mode: RGB")
                elif img.shape[2] == 4:
                    print("Mode: RGBA")
                else:
                    print(f"Unknown channel count: {img.shape[2]}")
            else:
                print(f"Unknown image shape: {img.shape}")
        else:
            if not isinstance(img, Image.Image):
                raise TypeError("Expected a PIL Image object for inspection")
            print(f"Format = {img.format}")
            print(f"Mode = {img.mode}")
            print(f"Size (W, H) = {img.size}")

    def show_image(self, img):

        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input must be a numpy array or PIL Image")

        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
        else:
            plt.imshow(np.array(img))
        plt.axis("off")
        plt.show()

    def compare_images(self, images_list, titles):
        if len(images_list) != len(titles):
            raise ValueError("Length of images_list and titles must be the same")

        n = len(images_list)
        plt.figure(figsize=(4 * n, 4))

        for i, (img, title) in enumerate(zip(images_list, titles)):
            plt.subplot(1, n, i + 1)
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    plt.imshow(img, cmap="gray")
                else:
                    plt.imshow(img)
            else:
                plt.imshow(np.array(img))
            plt.title(title)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def calculate_normal_difference_of_images(self, img1, img2):
        try:

            if img1 is None or img2 is None:
                raise ValueError("Input images cannot be None.")

            if not (hasattr(img1, "__getitem__") and hasattr(img2, "__getitem__")):
                raise TypeError("Inputs must be list-like or numpy arrays.")

            if len(img1) == 0 or len(img2) == 0:
                raise ValueError("Input images cannot be empty.")
            if len(img1) != len(img2) or len(img1[0]) != len(img2[0]):
                raise ValueError("Images must have the same dimensions.")

            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)
            N = len(img1) * len(img1[0])
            if N == 0:
                raise ValueError("Image size must be greater than zero.")

            return (
                np.sqrt(
                    np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
                )
                / 255.0
            )

        except Exception as e:
            print(f"Error in calculate_difference_of_images: {e}")
            return None
