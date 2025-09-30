import cv2
import os


class ImageIO:
    def __init__(self) -> None:
        pass

    def load_image(self, image_path, format="RGB"):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise IOError(f"Error opening image in {image_path}.")

        valid_formats = ["RGB", "RGBA", "HLS", "Grayscale", "LAB"]
        if format not in valid_formats:
            raise ValueError(
                f"Unsupported format: {format}. Choose from {valid_formats}"
            )
        if format == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif format == "Grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif format == "RGBA":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        elif format == "HLS":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif format == "LAB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return img

    def save_image(self, img, image_name: str, image_path="images/"):
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        try:
            cv2.imwrite(image_path + image_name, img)
        except Exception as e:
            raise IOError(f"Error saving image: {e}")
