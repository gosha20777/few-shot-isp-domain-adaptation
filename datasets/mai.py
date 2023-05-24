from tensorflow import keras
import numpy as np
from utils import preprocess

class MaiLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, 
            batch_size, 
            img_size, 
            dslr_scale, 
            input_img_paths, 
            target_img_paths
        ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.dslr_scale = dslr_scale
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        dslr_size = (self.img_size[0] * self.dslr_scale, self.img_size[1] * self.dslr_scale)

        x = np.zeros((self.batch_size,) + self.img_size + (4,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = preprocess.read_bayer_image(path)

        y = np.zeros((self.batch_size,) + dslr_size + (3,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            img = preprocess.read_target_image(path, dslr_size)
            y[j] = img
        
        return x, y