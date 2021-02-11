import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from data.dataset import MILImageDataset

class MILDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset_reference,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.train_sampler
        )

    def test_dataloader(self):
        return DataLoader(
            self.inference_dataset_reference,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

class LymphoDataModule(MILDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super(LymphoDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_tiles(row: pd.Series, phase: str):
        patient_id = row['id']
        patient_dir = Path(self.data_dir, f'{phase}', f'{patient_id}')
        return patient_dir.glob('*.jpg')

    def tile_dataframe(df: pd.DataFrame, phase: str):
        df['tiles'] = df.progress_apply(self.get_tiles, axis=1, phase=phase)
        return df.explode('tiles')

    def setup(self):
        
        tqdm.pandas()
        
        if Path(self.data_dir, 'train', 'train.csv').exists():
            print(f'Loading train data from file...')
            train_df = pd.read_csv(Path(self.data_dir, 'train', 'train.csv'))
            print(f'...done.')
        else:
            train_df = pd.read_csv(Path(self.data_dir, 'train', 'train_data.csv'))
            train_df = tile_dataframe(train_df, phase='train')
            train_df.to_csv(Path(self.data_dir, f'train.csv'))

        if Path(self.data_dir, f'test.csv').exists():
            print(f'Loading test slides from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test.csv'))
            print(f'...done.')
        else:
            test_df = pd.read_csv(Path(self.data_dir, 'test', 'test_data.csv'))
            test_df = tile_dataframe(train_df, phase='test')
            test_df.to_csv(Path(self.data_dir, f'test.csv'))

        train_df = train_df.sample(frac=0.005).reset_index()
        test_df = train_df.sample(frac=0.005).reset_index()
        self.train_dataset, self.test_dataset = (
            MILImageDataset(train_df, training=True),
            MILImageDataset(test_df, training=False)
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        return parser