import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, db_dir, batch_size, input_shape, num_classes,
                 shuffle=True):
        # TODO your initialization
        # you might want to store the parameters into class variables
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        # load the data from the root directory
        self.data, self.labels = self.get_data(db_dir)
        self.indices = np.arange(len(self.data))
        self.on_epoch_end()

    def get_data(self, root_dir):
        """"
        Loads the paths to the images and their corresponding labels from the database directory
        """
        # TODO your code here

        self.data = []
        self.labels = []

        for i in range(0, 70):
            if i < 10:
                folder = f"0{i}"
            else:
                folder = f"{i}"
            for j in range(0, 1000):
                img = folder
                if j < 10:
                    img += "0"
                if j < 100:
                    img += "0"
                img += f"{j}"
                imgPath = f"{root_dir}/{folder}000/{img}.png"
                self.data.append(imgPath)
                self.labels.append(img)

        return self.data, self.labels

    def __len__(self):
        """
        Returns the number of batches per epoch: the total size of the dataset divided by the batch size
        """
        return int(np.floor(len(self.data) / self.batch_size))
        # TODO your code  here (size of dataset divided by the batch size)

    def __getitem__(self, index):
        """"
        Generates a batch of data
        """
        batch_indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]
        batch_x = None # TODO load the image from batch_indices
        batch_y = None # TODO load the corresponding labels of the images you loaded
        # optionally you can use: batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        """"
        Called at the end of each epoch
        """
        # if required, shuffle your data after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            # TODO shuffle data
            # you might find np.random.shuffle useful here
            np.random.shuffle(self.indices)