import cv2
import numpy as np
import os


class Inspector:
    def __init__(self) -> None:
        pass

    def basic_inspector(
        self, img, is_verbose=True, want_size=False, image_path="images/image"
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

    pass
