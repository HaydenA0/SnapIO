# ./playgroud.py
from utilsio import ImageIO
import numpy as np
from visulization import Visualizer
from inspection import ImageInspector
from processing import Processor


io = ImageIO()
vis = Visualizer()
ii = ImageInspector()
proc = Processor()

img = io.load_image("./images/grass.png", format="Grayscale")
kernel = np.ones((3, 3)) / 9
