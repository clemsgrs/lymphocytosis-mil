import numpy as np
from torch.utils.data.sampler import Sampler


class TopKSampler(Sampler):
    def __init__(self, topk_indices, shuffle=True):
        self.indices = topk_indices
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
