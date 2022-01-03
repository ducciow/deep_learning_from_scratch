import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_size = image_size  # [height,width,channel]
        self.data_size = 100  # 100 images in total
        self.data_idx = 0  # generating image start from the 0-th
        self.num_epoch = 0
        self.end_of_data = False

        # read images with corresponding file names
        self.name_image_set = []  # [(name, img), (name, img), ...]
        for name in range(self.data_size):
            img_path = os.path.join(os.path.abspath(file_path), str(name) + '.npy')
            self.name_image_set.append((str(name), np.load(img_path)))
        # shuffle
        self.name_set, self.image_set = self._shuffle()
        # read labels
        self.label_set = json.loads(open(os.path.abspath(label_path)).read())

    def _shuffle(self):
        # shuffles the image set with corresponding names
        # return a tuple of numpy arrays
        #
        if self.shuffle:
            np.random.shuffle(self.name_image_set)
        names = []
        images = []
        for name, img in self.name_image_set:
            names.append(name)
            images.append(img)
        return np.array(names), np.array(images)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #
        if self.end_of_data:
            self.num_epoch += 1
            self.end_of_data = False
            if self.shuffle:
                self.name_set, self.image_set = self._shuffle()
        images = []
        labels = []
        names = []
        for i in range(self.batch_size):
            name = self.name_set[self.data_idx]
            img = self.image_set[self.data_idx]
            label = self.label_set[name]
            # perform mirroring or/and rotation
            img = self.augment(img)
            # resizing
            if self.image_size != img.shape:
                img = resize(img, self.image_size)
            images.append(img)
            labels.append(label)
            names.append(name)
            # update data_idx
            self.data_idx += 1
            if self.data_idx >= self.data_size:
                self.end_of_data = True
                self.data_idx %= self.data_size
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def _mirror(self, img):
        mirror_type = np.random.choice(4)  # 0: unchanged 1: horizontal 2: vertical 3: diagonal
        if mirror_type == 1:
            img_ = np.flip(img, 0)
        elif mirror_type == 2:
            img_ = np.flip(img, 1)
        elif mirror_type == 3:
            img_ = np.flip(img, (0, 1))
        else:
            img_ = img
        return img_

    def _rotate(self, img):
        rotate_degree = np.random.choice(4)  # 0: unchanged 1: 90 2: 180 3: 270
        img_ = np.rot90(img, rotate_degree)
        return img_

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            img = self._mirror(img)
        if self.rotation:
            img = self._rotate(img)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.num_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return self.class_dict[x]

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        images, labels = self.next()
        n_col = 3  # number of images shown per row
        n_row = int(np.ceil(self.batch_size / n_col))
        f, ax = plt.subplots(n_row, n_col)
        for r in range(n_row):
            for c in range(n_col):
                img_idx = r * n_col + c
                if img_idx < self.batch_size:
                    ax[r][c].set_title(self.class_name(labels[img_idx]))
                    ax[r][c].imshow(images[img_idx])
                ax[r][c].axis('off')
        plt.subplots_adjust(wspace=0.15, hspace=0.5)
        plt.show()
