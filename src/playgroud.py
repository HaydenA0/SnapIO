from utilsio import ImageIO
import cv2
from visulization import Visualizer
from inspection import ImageInspector
from processing import Processor


io = ImageIO()
vis = Visualizer()
ii = ImageInspector()
proc = Processor()

img = io.load_image("./images/grass.png", format="Grayscale")
vis.compare_images([img, 1 / (img + 1)], ["Image", "Image squared"])
print(proc.calculate_normal_difference_of_images(img, 1 / (img + 1)))
