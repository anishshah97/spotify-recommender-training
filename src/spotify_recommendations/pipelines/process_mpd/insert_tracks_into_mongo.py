import multiprocessing
import os

import dask.dataframe as dd
import dask.distributed as d_dist
import numpy as np
import pandas as pd
import pymongo
from dask.dataframe.core import repartition
from dask.distributed import Client
from dotenv import find_dotenv, load_dotenv
from pymongo import InsertOne, MongoClient, ReplaceOne

from .utils import chunk

num_cores = multiprocessing.cpu_count()
load_dotenv(find_dotenv())

#TODO: Generalize
# TODO: remove mongo from this to prevent a new connection per worker?
# TODO: multithreading inside of this mapped partition so each worker can efficiently process the chunk ops?


def upsert_mongo_data(df, id_col):

    MONGO_CONN = os.getenv("MONGO_CONN")
    mongo = MongoClient(MONGO_CONN)
    mongo_spotify_data = mongo["spotifyData"]
    mongo_spotify_track_features = mongo_spotify_data["trackFeatures"]
    mongo_spotify_artist_features = mongo_spotify_data["artistFeatures"]

    db_coll = mongo_spotify_track_features

    ops_list = []

    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        df[col] = df[col].fillna(pd.to_datetime('1970-01-01'))

    object_str_cols = df.select_dtypes(include=["string"]).columns
    for col in object_str_cols:
        df[col] = df[col].fillna("<NA>")

    feature_records = df.to_dict("records")
    for record in feature_records:
        ops_list.append(
            #             InsertOne({id_col: record[id_col]}, record)
            ReplaceOne({id_col: record[id_col]}, record, upsert=True)
        )
    chunked_ops = chunk(ops_list, 1000)
    for ops in chunked_ops:
        db_coll.bulk_write(ops, ordered=False)
    return True


def insert_tracks_into_mongo(mpd_track_features):
    MONGO_CONN = os.getenv("MONGO_CONN")

    mongo = MongoClient(MONGO_CONN)
    mongo_spotify_data = mongo["spotifyData"]
    mongo_spotify_track_features = mongo_spotify_data["trackFeatures"]
    mongo_spotify_artist_features = mongo_spotify_data["artistFeatures"]

    try:
        stored_tids = [doc.get("_id") for doc in mongo_spotify_track_features.aggregate([
            {"$group": {"_id": "$track_spid"}}
        ])]
    except:
        stored_tids = []

    client = d_dist.client._get_global_client() or Client()
    mpd_track_features["time_pulled"] = (
        mpd_track_features["time_pulled"]
        .fillna(pd.to_datetime('1970-01-01'))
        .map_partitions(
            pd.to_datetime, errors="coerce"
        )
    )

    mpd_track_features_df = mpd_track_features.compute()
    selected_mpd_track_features_df = mpd_track_features_df[~(
        mpd_track_features_df["track_spid"].isin(set(stored_tids)))]
    selected_mpd_tracks = dd.from_pandas(
        selected_mpd_track_features_df, npartitions=num_cores)
    _ = (
        selected_mpd_tracks
        #     mpd_track_features
        #     .repartition(npartitions=num_cores)
        .map_partitions(
            upsert_mongo_data,
            #         db_coll=mongo_mpd_track_features,
            id_col="track_spid",
            meta="float"
        )
    ).compute()

    return True
