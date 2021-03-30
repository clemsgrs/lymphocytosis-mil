import numpy as np
import pandas as pd
import torch
from typing import Tuple


class TopKProcessor:
    def __init__(self, topk_train: int = 1, topk_agg: int = 1, aggregation: str = 'max'):
        self.topk_train = topk_train
        self.topk_agg = topk_agg
        assert aggregation in ['max', 'mean']
        self.aggregation_func = np.amax if aggregation == 'max' else np.mean

    def __call__(
        self,
        df: pd.DataFrame,
        prob_col_name: str = 'prob',
        group: str = 'id'
    ) -> np.ndarray:
        
        topk_indices = np.hstack(df.groupby(group).apply(
            lambda gdf: gdf.sort_values(prob_col_name, ascending=False).index.values[:self.topk_train]
        ))
        return topk_indices

    def aggregate(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        prob_col_name: str = 'prob',
        group: str = 'id',
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # sub_df = df.loc[indices]
        grouped_sub_df = df.groupby(group)
        probs = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation_func(gdf.sort_values(prob_col_name, ascending=False)[prob_col_name].values[:self.topk_agg])).values
        )
        preds = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation_func(gdf.sort_values(prob_col_name, ascending=False)[prob_col_name].values[:self.topk_agg]) > threshold).values.astype('int')
        )
        labels = torch.from_numpy(grouped_sub_df.apply(
            lambda gdf: self.aggregation_func(gdf.label)).values.astype('int')
        )
        patient_ids = np.array(list(grouped_sub_df.groups.keys()))
        return patient_ids, probs, preds, labels