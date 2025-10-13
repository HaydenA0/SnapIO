from scipy.fft import fft2, fftshift
import numpy as np
import matplotlib.pyplot as plt


from utilsio import ImageIO
from visulization import Visualizer

io = ImageIO()
vis = Visualizer()

image = np.array(io.load_image("./images/grass.png", format="Grayscale"))

image_fourier_amplitudes = np.abs(np.array(fft2(image)))
visual_fourier_amplitudes = np.log(fftshift(image_fourier_amplitudes + 1))


vis.compare_images(
    [image, visual_fourier_amplitudes],
    ["Original Image", "$F(u,v)$ Fourier Transformation "],
)
