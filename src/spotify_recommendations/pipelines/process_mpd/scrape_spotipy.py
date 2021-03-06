import datetime
import math
import os
import time
from itertools import chain
from pathlib import Path

import dask.dataframe as dd
import dask.distributed as d_dist
import pandas as pd
import spotipy
from dask.distributed import Client
from loguru import logger

TRACKS_PER_CHUNK = 2500


def determine_path_for_features(**kwargs):
    data_path = Path(Path().resolve(), os.environ.get("DATA_DIR", "data"))
    mpd_features_path = Path(data_path, "03_primary", "mpd_track_features")
    mpd_features_path.mkdir(exist_ok=True)
    return mpd_features_path


def chunk(data, n):
    return [data[x: x + n] for x in range(0, len(data), n)]


def get_track_features(spotify, track_ids):
    time.sleep(0.25)
    if len(track_ids) > 100:
        logger.error("Too many tracks")
    else:
        return spotify.audio_features(track_ids)


def get_artist_features(spotify, artist_ids):
    time.sleep(0.25)
    if len(artist_ids) > 50:
        logger.error("Too many tracks")
    else:
        return spotify.artists(artist_ids)


def flatten_artist_features(artist_features):

    artist_follower_total = artist_features.get("followers", {}).get("total")
    artist_genres = artist_features.get("genres", [])
    artist_spid = artist_features.get("id")
    artist_img_urls = artist_features.get("images", [{"url": None}])
    if len(artist_img_urls) == 0:
        artist_img_url = None
    else:
        artist_img_url = artist_img_urls[0].get("url")
    artist_popularity = artist_features.get("popularity")

    flattened_artist_features = {
        "artist_follower_total": artist_follower_total,
        "artist_genres": artist_genres,
        "artist_spid": artist_spid,
        "artist_img_url": artist_img_url,
        "artist_popularity": artist_popularity,
    }

    return flattened_artist_features


def get_track_features_df(spotify, track_ids):

    chunked_track_ids = chunk(track_ids, 100)
    logger.info(f"Processing {len(chunked_track_ids)} chunks")
    chunked_track_features = [
        get_track_features(spotify, chunked_tracks)
        for chunked_tracks in chunked_track_ids
    ]
    track_features = [
        val for val in list(chain.from_iterable(chunked_track_features)) if val
    ]
    track_features_df = (
        pd.DataFrame(track_features)
        #         .drop(["uri", "track_href", "analysis_url", "type"], axis=1)
        .rename({"id": "spid"}, axis=1)
        .add_prefix("track_")
        # .set_index("track_spid")
    )
    track_features_df["time_pulled"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    logger.info(f"Returning track_features_df of size {track_features_df.shape}")
    return track_features_df


def get_artist_features_df(spotify, artist_ids):
    chunked_artist_ids = chunk(artist_ids, 50)
    logger.info(f"Processing {len(chunked_artist_ids)} chunks")
    chunked_artist_features = [
        get_artist_features(spotify, chunked_artists)["artists"]
        for chunked_artists in chunked_artist_ids
    ]
    artist_features = [
        val for val in list(chain.from_iterable(chunked_artist_features)) if val
    ]
    flattened_artist_features = [
        flatten_artist_features(artist) for artist in artist_features
    ]
    artist_features_df = pd.DataFrame(
        flattened_artist_features
    )  # .set_index("artist_spid")
    artist_features_df["artist_genres"] = artist_features_df["artist_genres"].map(list)
    artist_features_df["time_pulled"] = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat()
    logger.info(f"Returning artist_features_df of size {artist_features_df.shape}")
    return artist_features_df

# TODO: Pass kwargs bc no need for explicit reference to spotify needed here?


def gather_spotify_features_data(s, spotify, id_type="track", base_path=""):
    import uuid
    from pathlib import Path

    if id_type == "track":
        _feature_scraper = get_track_features_df
        # TODO: Better dtype checking/coeercion
        _meta = {
            'track_acousticness': "float",
            'track_analysis_url': "string",
            'track_danceability': "float",
            'track_duration_ms': "int",
            'track_energy': "float",
            'track_instrumentalness': "float",
            'track_key': "int",
            'track_liveness': "float",
            'track_loudness': "float",
            'track_mode': "int",
            'track_speechiness': "float",
            'track_spid': "string",
            'track_tempo': "float",
            'track_time_signature': "int",
            'track_track_href': "string",
            'track_type': "string",
            'track_uri': "string",
            'track_valence': "float"
        }
    elif id_type == "artist":
        _feature_scraper = get_artist_features_df
    else:
        raise ValueError(f"Improper ID type: {id_type}")

    _ids = s.unique().tolist()
    logger.info(f"Scraping {len(_ids)} {id_type}s")
    feature_df = _feature_scraper(spotify, _ids)
    time_pulled = datetime.datetime.now(datetime.timezone.utc).isoformat()
    feature_df["time_pulled"] = time_pulled
    for col, dtype in _meta.items():
        if col not in feature_df:
            feature_df[col] = pd.Series([], dtype=dtype)
        else:
            feature_df[col] = feature_df[col].astype(dtype)
    _df_id = str(uuid.uuid4().hex)
    feature_df.to_parquet(Path(base_path, _df_id + ".parquet"))

    return True


def dask_gather_spotify_features_data(ids_to_scrape, spotify, id_type="track", base_path=""):
    client = d_dist.client._get_global_client() or Client()
    logger.info("processing/cleaning")
    _ = ids_to_scrape.map_partitions(
        gather_spotify_features_data, spotify, id_type, base_path).compute().infer_objects()
    return True

# TODO: Remove dask from this and add a seperate preparation step?


def determine_remaining_ids_for_dask(dask_total_ids, dask_scraped_ids, seeded=True):
    mpd_ids = set(dask_total_ids.compute().unique())
    scraped_ids = set(dask_scraped_ids.compute().unique())
    remaining_ids = mpd_ids - scraped_ids
    remaining_ids_count = len(remaining_ids)
    logger.info(f"""
    Total Track Count: {len(mpd_ids)}
    Scraped Track Count: {len(scraped_ids)}
    Remaining Track Count: {remaining_ids_count}
    """)
    n_track_chunks = math.ceil(remaining_ids_count / TRACKS_PER_CHUNK)
    return dd.from_pandas(pd.Series(list(remaining_ids)), npartitions=n_track_chunks)


def get_spotify_credentials(**kwargs):
    auth_manager = spotipy.oauth2.SpotifyOAuth(
        scope="user-top-read%20user-read-currently-playing%20user-read-playback-state%20playlist-read-collaborative%20playlist-read-private%20user-library-read%20user-read-recently-played%20user-follow-read",
        show_dialog=True,
    )
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return spotify


def check_and_seed_feature_folder(mpd_track_ids, spotify, id_type="track", base_path=""):
    seed_id = determine_remaining_ids_for_dask(
        mpd_track_ids, dd.from_pandas(pd.Series([], dtype="string"), npartitions=1)).compute()[0:1]
    _ = gather_spotify_features_data(seed_id, spotify, id_type, base_path)
    return True
