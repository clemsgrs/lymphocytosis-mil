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


class LymphoDataModule():
    def __init__(self, data_dir: str, num_workers: int = 1, seed: int = 21):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed
    
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
            train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=self.seed)
            train_df = self.tile_dataframe(train_df, phase='train')
            val_df = self.tile_dataframe(val_df, phase='train')
            train_df.to_csv(Path(self.data_dir, f'train.csv'), index=False)
            val_df.to_csv(Path(self.data_dir, f'val.csv'), index=False)

        if Path(self.data_dir, f'test.csv').exists():
            print(f'Loading test slides from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
            print(f'...done.')
        else:
            test_df = pd.read_csv(Path(self.data_dir, 'test', 'test_data.csv'))
            test_df = self.tile_dataframe(test_df, phase='test')
            test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        train_df = train_df.reset_index()
        val_df = val_df.reset_index()
        self.train_dataset, self.val_dataset = (
            MILImageDataset(train_df, training=True),
            MILImageDataset(val_df, training=True)
        )


class TestDataModule():
    def __init__(self, data_dir: str, num_workers: int, seed: int = 21):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.seed = seed
    
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
            test_df.to_csv(Path(self.data_dir, f'test.csv'), index=False)

        self.test_dataset = (
            MILImageDataset(test_df, training=False)
        )