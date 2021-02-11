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


class DistributedTopKSampler(Sampler):
    """Some caveats here:
    - Separating by ID (i.e. slide)?
    - How to properly override just enough behavior of super class?
    See here for reference:
    https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
    """
    def __init__(self, *args, **kwargs):
        pass

    def __iter__(self):
        pass
