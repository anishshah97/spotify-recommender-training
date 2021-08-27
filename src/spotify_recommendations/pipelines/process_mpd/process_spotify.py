import dask.dataframe as dd
import dask.distributed as d_dist
from dask.dataframe.core import repartition
from dask.distributed import Client


def select_track_features(track_features):
    client = d_dist.client._get_global_client() or Client()
    desired_spotify_track_features = [
        "track_spid",
        "track_danceability",
        "track_energy",
        "track_key",
        "track_loudness",
        "track_mode",
        "track_speechiness",
        "track_acousticness",
        "track_instrumentalness",
        "track_liveness",
        "track_valence",
        "track_tempo",
    ]

    selected_mpd_track_features = track_features[desired_spotify_track_features]
    return selected_mpd_track_features
