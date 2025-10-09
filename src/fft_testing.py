from scipy.fft import fft2, fftshift, ifft2
import numpy as np

from utilsio import ImageIO
from visulization import Visualizer

io = ImageIO()
vis = Visualizer()


def fillter(image_fourier_amplitudes, amplitudes_range):
    h = np.zeros_like(image_fourier_amplitudes, dtype=int)
    h[image_fourier_amplitudes <= (image_fourier_amplitudes).max() / 2] = 1
    h[image_fourier_amplitudes >= (image_fourier_amplitudes).max() / 2] = 0
    return h


image = np.array(io.load_image("./images/tank.png", format="Grayscale"))

image_fourier = np.array(fft2(image))

image_fourier_amplitudes = np.abs(image_fourier)
amplitudes_range = np.max(image_fourier_amplitudes) - np.min(image_fourier_amplitudes)
H = fillter(image_fourier_amplitudes, amplitudes_range)

visual_fourier_amplitudes = np.log(fftshift(image_fourier_amplitudes + 1))

image_original = np.abs(np.array(ifft2(image_fourier * H)))

vis.compare_images(
    [image, visual_fourier_amplitudes, image_original],
    ["Image", "Transformation de fourier", "Original"],
)
print(H)
