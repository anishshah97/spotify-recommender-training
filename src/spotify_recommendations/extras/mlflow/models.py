import mlflow.pyfunc


class CosineModel(mlflow.pyfunc.PythonModel):

    def __init__(self, db):
        self.db = db

    def predict(self, ids, n=5):
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        db = self.db
        candidate_df = db[~(db.isin(set(ids)))]
        aggregate_query_feats = self.aggregate(ids)

        similarities = cosine_similarity(
            aggregate_query_feats.to_numpy().reshape(1, -1),
            candidate_df
        )
        top_n_indices = np.argpartition(similarities, -n, axis=1)[:, -n:]
        top_n_predictions = candidate_df.iloc[top_n_indices[0]].index.tolist()

        return top_n_predictions

    def aggregate(self, ids):
        db = self.db
        return db[db.index.isin(set(ids))].mean()

    def evaluate(self, ids, test_ids):
        n = len(test_ids)
        top_n_predictions = self.predict(ids, n)
        accuracy = len(set(top_n_predictions).intersection(test_ids)) / n
        return accuracy
