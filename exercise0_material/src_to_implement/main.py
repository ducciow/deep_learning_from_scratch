import numpy as np

from pattern import *
from generator import *

if __name__ == "__main__":
    #
    # TEST Checkerboard:

    # my_cb = Checker(20, 2)
    # my_cb.draw()
    # my_cb.show()

    #
    # TEST Circle:
    #
    # my_cc = Circle(128, 20, (80, 50))
    # my_cc.draw()
    # my_cc.show()

    # TEST Spectrum:
    #
    # my_spec = Spectrum(256)
    # my_spec.draw()
    # my_spec.show()

    #
    # TEST ImageGenerator:
    #
    g = ImageGenerator("./data/exercise_data", "./data/Labels.json", batch_size=12, image_size=[32, 32, 3],
                       rotation=False, mirroring=True, shuffle=True)
    g.show()
