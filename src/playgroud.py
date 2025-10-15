# ./playgroud.py
from utilsio import ImageIO
import numpy as np
from inspection import ImageInspector
from processing import Processor
from visulization import Visualizer


io = ImageIO()
ii = ImageInspector()
proc = Processor()
vis = Visualizer()

img = io.load_image("./images/grass.png", format="Grayscale")

img = np.array(img, dtype=np.float64)
kernel = np.ones((3, 3)) / 9


# Compare these 2
oldconv = proc.apply_kernel_old(img, kernel)
newconv = proc.apply_kernel_new(img, kernel)

vis.compare_images([img, newconv], ["Image", "Blurred"])
