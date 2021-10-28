import os

import dask.dataframe as dd
import dask.distributed as d_dist
# import mlflow
# import mlflow.pyfunc
import numpy as np
import pandas as pd
from dask.dataframe.core import repartition
from dask.distributed import Client
from loguru import logger
from tqdm import tqdm

# from spotify_recommendations.extras.mlflow.models import CosineModel

tqdm.pandas()


def save_pandas_cosine_model(selected_mpd_track_features, tid_col="track_spid"):
    # BUG: Remving custom class to deal with not importing custom module
    return CosineModel(selected_mpd_track_features.compute().set_index(tid_col))


def save_cosine_model_db(selected_mpd_track_features, tid_col="track_spid"):
    return selected_mpd_track_features.compute().set_index(tid_col)


def evaluate_cosine_model(cosine_model, split_mpd_pids_to_tids, row_limit=1):
    client = d_dist.client._get_global_client() or Client()
    needed_mpd_pids_to_tids_cats = split_mpd_pids_to_tids[["train", "test"]].compute()
    subset_needed_mpd_pids_to_tids_cats = needed_mpd_pids_to_tids_cats.sample(row_limit)
    all_scores = subset_needed_mpd_pids_to_tids_cats.progress_apply(
        lambda row: cosine_model._evaluate(row["train"], row["test"]), axis=1)
    logger.info(all_scores)
    return all_scores

# TODO: Make the format of the output nice here


def predict_w_db(db, data):
    # TODO: Remove hard coded cosine model class
    # BUG: workaround to deal with missing module
    class CosineModel(mlflow.pyfunc.PythonModel):

        def __init__(self, db):
            self.db = db

        def predict(self, model_input):
            import pandas as pd

            return model_input.apply(lambda row: self._predict(row["track_ids"], row["n"]), axis=1)

        def _predict(self, _ids, n=5):
            if isinstance(_ids, str):
                ids = eval(_ids)
            else:
                ids = _ids
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            db = self.db
            candidate_df = db[~(db.isin(set(ids)))]
            aggregate_query_feats = self.aggregate(ids)
            logger.info(aggregate_query_feats)

            similarities = cosine_similarity(
                aggregate_query_feats.to_numpy().reshape(1, -1),
                candidate_df
            )
            top_n_indices = np.argpartition(similarities, -n, axis=1)[:, -n:]
            top_n_predictions = candidate_df.iloc[top_n_indices[0]].index.tolist()
            logger.info(top_n_predictions)

            return top_n_predictions

        def aggregate(self, ids):
            db = self.db
            query_db = db[db.index.isin(set(ids))]
            mean_query = query_db.mean()
            return mean_query

        def _evaluate(self, ids, test_ids):

            n = len(test_ids)
            top_n_predictions = self._predict(ids, n)
            accuracy = len(set(top_n_predictions).intersection(test_ids)) / n
            return accuracy

    logger.info(data)
    model = CosineModel(db)
    prediction = model.predict(data)
    return prediction


def predict_w_mlflow(model, data):
    # BUG: have to deal with missing module

    logger.info(data)
    prediction = model.predict(data)
    return prediction

# TODO: Move to ETL


def create_inference_example(split_mpd_pids_to_tids):
    subset_pids_tids_df = split_mpd_pids_to_tids[["train"]].loc[0].compute()
    inference_example = subset_pids_tids_df.iloc[0:3]
    inference_example["n"] = 5
    inference_example = inference_example.rename({"train": "track_ids"}, axis=1)
    return inference_example
