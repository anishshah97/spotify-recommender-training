# import mlflow.pyfunc
# from loguru import logger


# class CosineModelWrapper(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         import pandas as pd

#         # load tokenizer and model from artifacts in model context
#         self.model = CosineModel(pd.read_parquet(context.artifacts["db.parquet"]))

#     def predict(self, context, model_input):
#         import pandas as pd

#         return self.model.predict(model_input)


# class CosineModel(mlflow.pyfunc.PythonModel):

#     def __init__(self, db):
#         self.db = db

#     def predict(self, model_input):
#         import pandas as pd

#         return model_input.apply(lambda row: self._predict(row["track_ids"], row["n"]), axis=1)

#     def _predict(self, _ids, n=5):
#         if isinstance(_ids, str):
#             ids = eval(_ids)
#         else:
#             ids = _ids
#         import numpy as np
#         from sklearn.metrics.pairwise import cosine_similarity

#         db = self.db
#         candidate_df = db[~(db.isin(set(ids)))]
#         aggregate_query_feats = self.aggregate(ids)
#         logger.info(aggregate_query_feats)

#         similarities = cosine_similarity(
#             aggregate_query_feats.to_numpy().reshape(1, -1),
#             candidate_df
#         )
#         top_n_indices = np.argpartition(similarities, -n, axis=1)[:, -n:]
#         top_n_predictions = candidate_df.iloc[top_n_indices[0]].index.tolist()
#         logger.info(top_n_predictions)

#         return top_n_predictions

#     def aggregate(self, ids):
#         db = self.db
#         query_db = db[db.index.isin(set(ids))]
#         mean_query = query_db.mean()
#         return mean_query

#     def _evaluate(self, ids, test_ids):

#         n = len(test_ids)
#         top_n_predictions = self._predict(ids, n)
#         accuracy = len(set(top_n_predictions).intersection(test_ids)) / n
#         return accuracy
