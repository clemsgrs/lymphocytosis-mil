import pandas as pd
from typing import Union


def get_valid_tiles(row: pd.Series, t_x: int = 32, t_y: int = 32):
    if isinstance(row.image_dimensions, tuple):
        max_y, max_x = row.image_dimensions
    elif isinstance(row.image_dimensions, str):
        max_y, max_x = list(map(int, row.image_dimensions[1:-1].split(',')))
    coords = []
    for y in range(0, max_y, t_y):
        for x in range(0, max_x, t_x):
            coords.append((y, x))
    return coords


def tile_dataframe(df: pd.DataFrame, tile_size: Union[int, tuple]):
    if isinstance(tile_size, int):
        t_x, t_y = tile_size, tile_size
    elif isinstance(tile_size, tuple):
        t_x, t_y = tile_size
    df['tiles'] = df.progress_apply(get_valid_tiles, axis=1, t_x=t_x, t_y=t_y)
    return df.explode('tiles').rename(columns={'tiles': 'coord'})