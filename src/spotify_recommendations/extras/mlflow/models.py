import mlflow.pyfunc
from loguru import logger


class CosineModel(mlflow.pyfunc.PythonModel):

    def __init__(self, db):
        self.db = db

    def predict(self, query_df):
        return query_df.apply(lambda row: self._predict(row["track_ids"], row["n"]), axis=1)

    def _predict(self, ids, n=5):
        logger.info(ids)
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
