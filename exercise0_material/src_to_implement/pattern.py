import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        # check if resolution is evenly dividable by 2 * tile_size
        assert (resolution / tile_size * 0.5) % 1 == 0, "INVALID INPUT"
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # a tile block contains 4 tiles
        tile_block = np.zeros((self.tile_size * 2, self.tile_size * 2))
        tile_block[:self.tile_size, self.tile_size:] += 1
        tile_block[self.tile_size:, :self.tile_size] += 1

        n_tiles = self.resolution // self.tile_size  # number of tiles per row/column
        self.output = np.tile(tile_block, (n_tiles // 2, n_tiles // 2))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.x, self.y = position
        self.output = np.zeros((resolution, resolution))

    def draw(self):
        coordinate_values = np.arange(self.resolution)  # the max value along axis, i.e. resolution
        x_values, y_values = np.meshgrid(coordinate_values, coordinate_values)  # two 2-d arrays for plot
        # mask the circle area
        within_circle = np.square(x_values - self.x) + np.square(y_values - self.y) <= np.square(self.radius)
        self.output[within_circle] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.empty((self.resolution, self.resolution, 3))

    def draw(self):
        # red increasing from left to right horizontally
        self.output[:, :, 0] = np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1))
        # green increasing from top to bottom vertically
        self.output[:, :, 1] = np.repeat(np.linspace(0, 1, self.resolution), self.resolution) \
            .reshape((self.resolution, self.resolution))
        # blue increasing from right to left horizontally
        self.output[:, :, 2] = np.tile(np.linspace(1, 0, self.resolution), (self.resolution, 1))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.axis('off')
        plt.show()
