import os

import dask.dataframe as dd
import dask.distributed as d_dist
import mlflow
import numpy as np
import pandas as pd
from dask.dataframe.core import repartition
from dask.distributed import Client
from loguru import logger
from tqdm import tqdm

from spotify_recommendations.extras.mlflow.models import CosineModel

tqdm.pandas()


def save_pandas_cosine_model(selected_mpd_track_features):
    return CosineModel(selected_mpd_track_features)


def evaluate_cosine_model(cosine_model, split_mpd_pids_to_tids, row_limit=10):
    client = d_dist.client._get_global_client() or Client()
    needed_mpd_pids_to_tids_cats = split_mpd_pids_to_tids[["train", "test"]].compute()
    subset_needed_mpd_pids_to_tids_cats = needed_mpd_pids_to_tids_cats.sample(row_limit)
    all_scores = subset_needed_mpd_pids_to_tids_cats.progress_apply(
        lambda row: cosine_model._evaluate(row["train"], row["test"]), axis=1)
    logger.info(all_scores)
    return all_scores
