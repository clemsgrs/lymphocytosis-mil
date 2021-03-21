from functools import lru_cache
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from typing import Callable, Union


@lru_cache(maxsize=256)
def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class MILImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        training: bool = True,
        transform: Callable = None
    ):
        self.dataset = dataset
        self.training = training
        self.transform = transform

    @lru_cache(maxsize=4096)
    def __getitem__(self, index: int):
        row = self.dataset.loc[index]
        image_fp = row.tiles
        image = read_image(image_fp)
        lymph_count = np.array([row.lymph_count]).astype(float)
        label = np.array([row.label]).astype(float) if self.training else np.array([-1])
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.functional.to_tensor(image)
        return index, image, lymph_count, label

    def __len__(self):
        return len(self.dataset)
