"""
Module for loading custom Cosine Similarity model using pandas as a pseduo db
"""
import os
from copy import deepcopy
from distutils.dir_util import copy_tree
# TODO: Rename as this can be used for general joblib/pickle loading
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from loguru import logger
from mlflow import pyfunc
from mlflow.exceptions import MlflowException

from .models import CosineModel, CosineModelWrapper

FLAVOR_NAME = "cosine_model"


def get_requirements():
    from sys import version_info

    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                      minor=version_info.minor,
                                                      micro=version_info.micro)
    import numpy
    import pandas
    import pyarrow
    import sklearn

    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            'pip',
            {
                'pip': [
                    'mlflow',
                    'pandas=={}'.format(pandas.__version__),
                    'pyarrow=={}'.format(pyarrow.__version__),
                    'numpy=={}'.format(numpy.__version__),
                    'scikit-learn=={}'.format(sklearn.__version__),
                ],
            },
        ],
        'name': 'cosine_model'
    }
    return conda_env


def save_model(
    cosine_model,
    path,
    **kwargs
):

    # BUG: Creates extra folder in the same directory as model for now?
    path = Path(os.path.abspath(path))
    # output_date = datetime.datetime.now().strftime("%Y-%m-%dT%H.%m.%sZ")
    # ts_path = Path(path, output_date)
    # main_model_path = Path(ts_path, "cosine_model")
    # main_model_path = Path(path, "cosine_model")
    # if os.path.exists(main_model_path):
    #     raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "db.parquet"
    # model_data_path = Path(ts_path, model_data_subpath).resolve()
    model_data_path = Path(path.parent, model_data_subpath).resolve()
    # os.makedirs(ts_path)

    cosine_model.db.head(5).to_parquet(model_data_path)
    artifacts = {
        "db.parquet": model_data_subpath
    }

    conda_env = get_requirements()

    pyfunc.save_model(
        path=path,
        python_model=CosineModelWrapper(),
        code_path=["models.py"],
        artifacts=artifacts,
        conda_env=conda_env)


# def log_model(
#     cosine_model,
#     artifact_path,
#     **kwargs
# ):
#     conda_env = get_requirements()
#     pyfunc.log_model(artifact_path=artifact_path,
#                      python_model=CosineModel, conda_env=conda_env)
