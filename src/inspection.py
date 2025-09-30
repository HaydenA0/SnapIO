import cv2
import numpy as np
import os
import math


class ImageInspector:

    def _validate_image(self, img: np.ndarray) -> None:
        if not isinstance(img, np.ndarray):
            raise TypeError("Input image must be a NumPy array.")
        if img.ndim not in [2, 3]:
            raise ValueError(
                f"Input image must be 2D (grayscale) or 3D (color), but got {img.ndim} dimensions."
            )

    def get_basic_properties(
        self, img: np.ndarray, image_path: str = "images/", is_verbose: bool = True
    ) -> dict:
        self._validate_image(img)

        if img.ndim == 2:
            height, width = img.shape
            channels = 1
        else:
            height, width, channels = img.shape

        encoding = img.dtype
        bytes_per_channel = img.itemsize
        size_in_memory = img.nbytes

        try:

            divisor = math.gcd(width, height)
            aspect_ratio = (width // divisor, height // divisor)
        except Exception:

            aspect_ratio = (width, height)

        properties = {
            "height": height,
            "width": width,
            "channels": channels,
            "encoding": encoding,
            "bytes_per_channel": bytes_per_channel,
            "size_in_memory_bytes": size_in_memory,
            "aspect_ratio": f"{aspect_ratio[0]}:{aspect_ratio[1]}",
        }

        if image_path:
            try:
                properties["size_on_disk_bytes"] = os.path.getsize(image_path)
            except FileNotFoundError:
                print(
                    f"Warning: Could not find file at '{image_path}' to get disk size."
                )
            except Exception as e:
                print(
                    f"Warning: Could not get disk size for '{image_path}'. Error: {e}"
                )

        if is_verbose:
            print("--- Image Basic Properties ---")
            for key, value in properties.items():
                print(f"{key.replace('_', ' ').title():<25}: {value}")
            print("----------------------------")

        return properties

    def get_statistics(self, img: np.ndarray) -> dict:
        self._validate_image(img)

        if img.ndim == 2:
            mean, std_dev = cv2.meanStdDev(img)
            min_val, max_val, _, _ = cv2.minMaxLoc(img)
            median = np.median(img)
            stats = {
                "mean": mean[0][0],
                "std_dev": std_dev[0][0],
                "min": min_val,
                "max": max_val,
                "median": median,
            }
        else:

            mean, std_dev = cv2.meanStdDev(img)
            b, g, r = cv2.split(img)

            min_vals = [cv2.minMaxLoc(c)[0] for c in (b, g, r)]
            max_vals = [cv2.minMaxLoc(c)[1] for c in (b, g, r)]
            medians = [np.median(c) for c in (list(b), list(g), list(r))]

            stats = {
                "mean_bgr": mean.flatten().tolist(),
                "std_dev_bgr": std_dev.flatten().tolist(),
                "min_bgr": min_vals,
                "max_bgr": max_vals,
                "median_bgr": medians,
            }
        return stats

    def get_content_analysis(self, img: np.ndarray) -> dict:
        self._validate_image(img)
        if img.dtype != np.uint8:
            raise TypeError("Content analysis requires a uint8 image (0-255 range).")

        if img.ndim == 3:

            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = img

        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist_normalized = hist.ravel() / hist.sum()

        entropy = -np.sum(
            hist_normalized * np.log2(hist_normalized + np.finfo(float).eps)
        )

        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        return {
            "shannon_entropy": entropy,
            "laplacian_variance (blur)": laplacian_var,
            "normalized_histogram": hist_normalized,
        }
