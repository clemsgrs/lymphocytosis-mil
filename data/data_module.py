import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from core.data.dataset import MILImageDataset
from core.data.utils import tile_dataframe


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


class PCAMDataModule(MILDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, tile_size: int):
        super(PCAMDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tile_size = tile_size

    def setup(self, stage: Optional[str] = None):
        tqdm.pandas()
        if Path(self.data_dir, f'train_slides.csv').exists():
            print(f'Loading train slides from file...')
            train_df = pd.read_csv(Path(self.data_dir, f'train_slides.csv'))
            train_df['image_dimensions'] = train_df['image_dimensions'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            print(f'...done.')
        else:
            train_df = pd.read_csv(Path(self.data_dir, 'train_labels.csv'))
            train_df['image_fp'] = train_df.id.apply(
                lambda p: Path(self.data_dir, 'train', f'{p}.tif')
            )
            train_df['image_dimensions'] = train_df.image_fp.progress_apply(
                lambda p: Image.open(p).size
            )
            train_df.to_csv(Path(self.data_dir, f'train_slides.csv'))

        if Path(self.data_dir, f'test_slides.csv').exists():
            print(f'Loading test slides from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test_slides.csv'))
            test_df['image_dimensions'] = test_df['image_dimensions'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            print(f'...done.')
        else:
            test_slides = list(Path(self.data_dir, 'test').glob('*.tif'))
            test_df = pd.DataFrame({'id': [p.stem for p in test_slides]})
            test_df['image_fp'] = test_df.id.apply(
                lambda p: Path(self.data_dir, 'test', f'{p}.tif')
            )
            test_df['image_dimensions'] = test_df.image_fp.progress_apply(
                lambda p: Image.open(p).size
            )
            test_df.to_csv(Path(self.data_dir, f'test_slides.csv'))

        if Path(self.data_dir, f'train_tiles_{str(self.tile_size)}.csv').exists():
            print(f'Loading train tiles from file...')
            train_df = pd.read_csv(Path(self.data_dir, f'train_tiles_{str(self.tile_size)}.csv'))
            train_df['image_dimensions'] = train_df['image_dimensions'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            train_df['coord'] = train_df['coord'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            print(f'...done.')
        else:
            train_df = tile_dataframe(train_df, self.tile_size)
            train_df.to_csv(Path(self.data_dir, f'train_tiles_{str(self.tile_size)}.csv'))
        if Path(self.data_dir, f'test_tiles_{str(self.tile_size)}.csv').exists():
            print(f'Loading test tiles from file...')
            test_df = pd.read_csv(Path(self.data_dir, f'test_tiles_{str(self.tile_size)}.csv'))
            test_df['image_dimensions'] = test_df['image_dimensions'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            test_df['coord'] = test_df['coord'].apply(
                lambda dim: tuple(list(map(int, dim[1:-1].split(','))))
            )
            print(f'...done.')
        else:
            test_df = tile_dataframe(test_df, self.tile_size)
            test_df.to_csv(Path(self.data_dir, f'test_tiles_{str(self.tile_size)}.csv'))

        train_df = train_df.sample(frac=0.005).reset_index()
        test_df = train_df.sample(frac=0.005).reset_index()
        self.train_dataset, self.test_dataset = (
            MILImageDataset(train_df, self.tile_size, training=True),
            MILImageDataset(test_df, self.tile_size, training=False)
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--tile_size', type=int, default=32)
        return parser