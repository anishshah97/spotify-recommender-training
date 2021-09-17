from kedro.pipeline import Pipeline, node

from .clean_mpd import multiprocess_clean_mpd
from .cosine_experiment import (
    create_inference_example,
    evaluate_cosine_model,
    predict_w_db,
    predict_w_mlflow,
    save_cosine_model_db,
    save_pandas_cosine_model,
)
from .dask_utils import dask_dedup
from .insert_tracks_into_mongo import insert_tracks_into_mongo
from .mpd_playlist_features import select_mpd_features
from .obfuscate_mpd_playlist import obfuscate_mpd
from .process_spotify import select_track_features
from .scrape_spotipy import (
    check_and_seed_feature_folder,
    dask_gather_spotify_features_data,
    determine_path_for_features,
    determine_remaining_ids_for_dask,
    get_spotify_credentials,
)


def prepare_mpd_dataset(**kwargs):
    return Pipeline([
        node(obfuscate_mpd, "cleaned_mpd_playlists",
             "split_mpd_pids_to_tids", tags=["training"]),
        node(select_track_features, "mpd_track_features",
             "selected_mpd_track_features", tags=["training"]),
        node(select_mpd_features, "cleaned_mpd_playlists",
             "selected_mpd_playlist_features", tags=["training"]),
        node(create_inference_example, ["split_mpd_pids_to_tids"],
             "inference_example", tags=["training"])
    ])


def scrape_spotify_for_mpd(**kwargs):
    return Pipeline([
        node(
            multiprocess_clean_mpd, [], "mpd_cleaning_notification", tags=["etl"]
        ),
        node(dask_dedup, ["cleaned_mpd_tracks", "params:track_id_col", "mpd_cleaning_notification"],
             "deduped_cleaned_mpd_tracks", tags=["etl"]),
        node(get_spotify_credentials, [], "spotify", tags=["etl"]),
        node(determine_path_for_features, [], "mpd_features_path", tags=["etl"]),
        node(check_and_seed_feature_folder, [
             "mpd_track_ids", "spotify", "params:track_feature_type", "mpd_features_path"],
             "track_feature_seed_notification", tags=["etl"]),
        node(determine_remaining_ids_for_dask, [
             "mpd_track_ids", "scraped_mpd_track_feature_ids", "track_feature_seed_notification"],
             "dd_remaining_ids", tags=["etl"]),
        node(dask_gather_spotify_features_data, [
             "dd_remaining_ids", "spotify", "params:track_feature_type", "mpd_features_path"],
             "track_scraping_notification", tags=["etl"])
    ])


def insert_into_mongo(**kwargs):
    return Pipeline([
        node(insert_tracks_into_mongo, "deduped_cleaned_mpd_tracks",
             "insertion_complete_notification", tags=["etl"])
    ])


def perform_cosine_experiment(**kwargs):
    return Pipeline([
        node(save_pandas_cosine_model, "selected_mpd_track_features",
             "cosine_model", tags=["training"]),
        node(evaluate_cosine_model, ["cosine_model",
                                     "split_mpd_pids_to_tids"], "all_scores", tags=["training", "evaluation"]),
        node(predict_w_mlflow, ["cosine_model", "inference_example"],
             "prediction", tags=["training", "inference"])
    ])


def ml_pipeline(**kwargs):
    return prepare_mpd_dataset() + perform_cosine_experiment()
