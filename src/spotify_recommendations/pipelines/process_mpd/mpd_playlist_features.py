from datetime import datetime

import dask.dataframe as dd
import dask.distributed as d_dist
from dask.dataframe.core import repartition
from dask.distributed import Client

from .utils import convert_text_bool_to_python_bool


def select_mpd_features(cleaned_mpd_playlists):
    client = d_dist.client._get_global_client() or Client()
    playlist_feature_cols = [
        "pid",
        "name",
        "collaborative",
        "modified_at",
        "num_followers",
        "num_edits"
    ]
    cleaned_mpd_playlists["collaborative"] = (
        cleaned_mpd_playlists["collaborative"]
        .map(convert_text_bool_to_python_bool)
    )
    cleaned_mpd_playlists["modified_at"] = (
        cleaned_mpd_playlists["modified_at"]
        .map(datetime.utcfromtimestamp)
    )
    selected_clean_playlists = cleaned_mpd_playlists[playlist_feature_cols]
    return selected_clean_playlists
