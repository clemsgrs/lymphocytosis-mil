import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from functools import lru_cache
from torchvision import transforms
from typing import Callable, Union
from sklearn.model_selection import train_test_split


@lru_cache(maxsize=256)
def read_image(image_fp: str) -> Image:
    return Image.open(image_fp)


class MILImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        training: bool = True,
        transform: Callable = None
    ):
        self.df = df
        self.training = training
        self.transform = transform

    @lru_cache(maxsize=4096)
    def __getitem__(self, index: int):
        row = self.df.loc[index]
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
        return len(self.df)


class LymphoDataModule():
    def __init__(self, data_dir: str, num_workers: int = 1, val_size: float = 0.3, pct: bool = False, seed: int = 21, save_csv: bool = False):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.val_size = val_size
        self.pct = pct
        self.seed = seed
        self.save_csv = save_csv
    
    def get_tiles(self, row: pd.Series, phase: str):
        patient_id = row['id']
        patient_dir = Path(self.data_dir, f'{phase}', f'{patient_id}')
        return list(patient_dir.glob('*.jpg'))

    def tile_dataframe(self, df: pd.DataFrame, phase: str):
        df['tiles'] = df.progress_apply(self.get_tiles, axis=1, phase=phase)
        return df.explode('tiles')

    def setup(self):
        
        tqdm.pandas()
        
        if Path(self.data_dir, 'train.csv').exists() and Path(self.data_dir, 'val.csv').exists():
            print(f'Loading train data from file...')
            train_df = pd.read_csv(Path(self.data_dir, 'train.csv'))
            print(f'...done.')
            print(f'Loading validation data from file...')
            val_df = pd.read_csv(Path(self.data_dir, 'val.csv'))
            print(f'...done.')
        else:
            train_df = pd.read_csv(Path(self.data_dir, 'train', 'train_data.csv'))
            train_df, val_df = train_test_split(train_df, test_size=self.val_size, random_state=self.seed)
            train_df = self.tile_dataframe(train_df, phase='train')
            val_df = self.tile_dataframe(val_df, phase='train')
            if self.save_csv:
                train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
                val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        if self.pct:
            train_df = train_df.sample(frac=0.01).reset_index()
            val_df = val_df.sample(frac=0.01).reset_index()
        else:
            train_df = train_df.reset_index()
            val_df = val_df.reset_index()
        self.train_dataset, self.val_dataset = (
            MILImageDataset(train_df, training=True),
            MILImageDataset(val_df, training=True)
        )


class TestDataModule():
    def __init__(self, data_dir: str, num_workers: int = 1, save_csv: bool = False):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.save_csv = save_csv
    
    def get_tiles(self, row: pd.Series, phase: str):
        patient_id = row['id']
        patient_dir = Path(self.data_dir, f'{phase}', f'{patient_id}')
        return list(patient_dir.glob('*.jpg'))

    def tile_dataframe(self, df: pd.DataFrame, phase: str):
        df['tiles'] = df.progress_apply(self.get_tiles, axis=1, phase=phase)
        return df.explode('tiles')

    def setup(self):
        
        tqdm.pandas()

        if Path(self.data_dir, f'test.csv').exists():
            print(f'Loading test slides from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
            print(f'...done.')
        else:
            test_df = pd.read_csv(Path(self.data_dir, 'test', 'test_data.csv'))
            test_df = self.tile_dataframe(test_df, phase='test')
            if self.save_csv:
                test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        self.test_dataset = (
            MILImageDataset(test_df, training=False)
        )