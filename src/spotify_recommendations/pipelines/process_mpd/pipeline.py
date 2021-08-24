from kedro.pipeline import Pipeline, node

from .clean_mpd import multiprocess_clean_mpd
from .dask_utils import dask_dedup
from .scrape_spotipy import (
    check_and_seed_feature_folder,
    dask_gather_spotify_features_data,
    determine_path_for_features,
    determine_remaining_ids_for_dask,
    get_spotify_credentials,
)


def prepare_mpd_dataset(**kwargs):
    return Pipeline([
        node(
            multiprocess_clean_mpd, [], "mpd_cleaning_notification"
        ),
        node(dask_dedup, ["cleaned_mpd_tracks", "params:track_id_col", "mpd_cleaning_notification"],
             "deduped_cleaned_mpd_tracks"),

    ])


def scrape_spotify_for_mpd(**kwargs):
    return Pipeline([
        node(get_spotify_credentials, [], "spotify"),
        node(determine_path_for_features, [], "mpd_features_path"),
        node(check_and_seed_feature_folder, [
             "mpd_track_ids", "spotify", "params:track_feature_type", "mpd_features_path"], "track_feature_seed_notification"),
        node(determine_remaining_ids_for_dask, [
             "mpd_track_ids", "scraped_mpd_track_feature_ids", "track_feature_seed_notification"], "dd_remaining_ids"),
        node(dask_gather_spotify_features_data, [
             "dd_remaining_ids", "spotify", "params:track_feature_type", "mpd_features_path"], "track_scraping_notification")
    ])
