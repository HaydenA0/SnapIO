# ./playgroud.py
from utilsio import ImageIO
import numpy as np
from inspection import ImageInspector
from processing import Processor


io = ImageIO()
ii = ImageInspector()
proc = Processor()

img = io.load_image("./images/grass.png", format="Grayscale")
img = np.array(img, dtype=np.float64)
kernel = np.ones((3, 3)) / 9


# Compare these 2
oldconv = proc.apply_kernel_old(img, kernel)
newconv = proc.apply_kernel_new(img, kernel)
