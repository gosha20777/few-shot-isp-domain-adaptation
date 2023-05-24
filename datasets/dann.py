from tensorflow import keras
import numpy as np
from utils import preprocess
import random

class DannLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, 
            source_loader,
            target_loader
        ):
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.source_len = len(source_loader)
        self.target_len = len(target_loader)
        self.min_len = min(self.source_len, self.target_len)
        self.step = max(self.source_len, self.target_len) // self.min_len
        self.batch_size = source_loader.batch_size

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.step + random.randint(0, self.min_len - 1)

        if self.source_len > self.target_len:
            s_x, s_y = self.source_loader.__getitem__(i)
            t_x, t_y = self.target_loader.__getitem__(idx)
        else:
            s_x, s_y = self.source_loader.__getitem__(idx)
            t_x, t_y = self.target_loader.__getitem__(i)

        d_s_y = np.zeros(shape=(self.batch_size, 1))
        d_t_y = np.ones(shape=(self.batch_size, 1))
        
        x = (s_x, t_x)
        y = (s_y, t_y, d_s_y, d_t_y)
        
        return x, y 