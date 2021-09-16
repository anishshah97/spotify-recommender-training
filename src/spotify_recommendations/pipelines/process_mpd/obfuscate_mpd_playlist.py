import dask.dataframe as dd
import dask.distributed as d_dist
from dask.dataframe.core import repartition
from dask.distributed import Client
from loguru import logger
from sklearn.model_selection import train_test_split


def obfuscate_mpd(cleaned_mpd_playlists):
    client = d_dist.client._get_global_client() or Client()
    logger.info(cleaned_mpd_playlists.columns)

    pids_to_tids = cleaned_mpd_playlists[["pid", "track_ids"]].set_index("pid")[
        "track_ids"]
    # We do not care for order mattering here as of now
    # There is no user information so our task will focus on recommending songs to playlists
    #  with expansion into optimizing ordering or playlists and then further playlist continuation where sequence matters
    split_mpd_pids_to_tids = pids_to_tids.map(train_test_split).reset_index()
    split_mpd_pids_to_tids = split_mpd_pids_to_tids.assign(train=split_mpd_pids_to_tids["track_ids"].map(lambda x: x[0]),
                                                           test=split_mpd_pids_to_tids["track_ids"].map(lambda x: x[1]))
    return split_mpd_pids_to_tids
