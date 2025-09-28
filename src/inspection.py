from os.path import normpath
import cv2
import numpy as np
import os


def calculate_luminance(normalized_image):
    channel = normalized_image.shape[-1]
    if channel == 3:
        red = normalized_image[:, :, 0]
        green = normalized_image[:, :, 1]
        blue = normalized_image[:, :, 2]
        luminance_image = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        return (
            np.mean(luminance_image),
            np.min(luminance_image),
            np.max(luminance_image),
        )
    else:
        return (
            np.mean(normalized_image),
            np.min(normalized_image),
            np.max(normalized_image),
        )


def calculate_histogram(normalized_image):
    histo = np.histogram(normalized_image, bins=256, density=True)[0]
    return histo


class Inspector:
    def __init__(self) -> None:
        pass

    def basic_inspector(
        self,
        img,
        is_verbose=True,
        want_size=False,
        image_path="images/image",
    ):
        height, width, channels = img.shape
        if want_size:
            try:
                image_size_on_disk = os.path.getsize(image_path)
                if is_verbose:
                    print(f"Image size on disk is {image_size_on_disk}")
            except Exception as e:
                raise ValueError(f"Couldn't get image size, check path.")

        encoding = img.dtype
        bytes_per_channel = img.itemsize
        image_size_on_memory = height * width * channels * bytes_per_channel
        divisor = np.gcd(width, height)
        aspect_ratio = (width // divisor, height // divisor)

        if is_verbose:
            print(
                f"Image height : {height}\n Image width : {width}\n Image channels : {channels}"
            )
            print(
                f"Image Encoding : {encoding}\n Image bytes per channel : {bytes_per_channel}"
            )
            print(
                f"Image size on RAM : {image_size_on_memory}\n Image aspect_ratio is {aspect_ratio[0]}:{aspect_ratio[1]}"
            )
        return (
            height,
            width,
            channels,
            encoding,
            bytes_per_channel,
            image_size_on_memory,
            aspect_ratio,
        )

    def basic_statistcs(self, img):
        channel = img.shape[-1]
        if channel == 1:
            mean = np.mean(img)
            median = np.median(img)
            pmin = np.min(img)
            pmax = np.max(img)
            std_dev = np.std(img)
            return {
                "mean": mean,
                "median": median,
                "min": pmin,
                "max": pmax,
                "std_dev": std_dev,
            }

        else:

            means = []
            medians = []
            pmins = []
            pmaxs = []
            std_devs = []

            for i in range(channel):
                single_channel = img[:, :, i]
                means.append(np.mean(single_channel))
                medians.append(np.median(single_channel))
                pmins.append(np.min(single_channel))
                pmaxs.append(np.max(single_channel))
                std_devs.append(np.std(single_channel))

            return {
                "mean": means,
                "median": medians,
                "min": pmins,
                "max": pmaxs,
                "std_dev": std_devs,
            }

    def basic_analysis(self, img):
        normalized_image = np.array(img / 255.0, dtype=np.float32)
        histo = calculate_histogram(normalized_image)
        shannon_contrast = -np.sum(histo * np.log2(histo))
        luminance_info = calculate_luminance(normalized_image)
        grey_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(grey_image, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        return {
            "Histogram Noramlized": histo,
            "Contrast using Entropy": shannon_contrast,
            "Luminance Information": luminance_info,
            "Laplacien Variance": laplacian_variance,
        }
