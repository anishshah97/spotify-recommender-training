# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project pipelines."""
from sys import version_info
from typing import Dict

import kedro
import kedro_mlflow
import numpy
import pandas
import sklearn
from kedro.pipeline import Pipeline
from kedro_mlflow.pipeline import pipeline_ml_factory

from .pipelines import process_mpd as mpd


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    prepare_mpd = mpd.prepare_mpd_dataset()
    scrape_spotify_for_mpd = mpd.scrape_spotify_for_mpd()
    insert_into_mongo = mpd.insert_into_mongo()
    perform_cosine_experiment = mpd.perform_cosine_experiment()
    ml_pipeline = mpd.ml_pipeline()

    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                      minor=version_info.minor,
                                                      micro=version_info.micro)

    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            'python={}'.format(PYTHON_VERSION),
            'pip',
            {
                'pip': [
                    'mlflow',
                    'pandas=={}'.format(pandas.__version__),
                    'numpy=={}'.format(numpy.__version__),
                    'scikit-learn=={}'.format(sklearn.__version__),
                    'kedro=={}'.format(kedro.__version__),
                    'kedro-mlflow=={}'.format(kedro_mlflow.__version__)
                ],
            },
        ],
        'name': 'cosine_model'
    }

    mpd_cosine_ml = pipeline_ml_factory(
        training=ml_pipeline.only_nodes_with_tags("training"),
        inference=ml_pipeline.only_nodes_with_tags("inference"),
        input_name="inference_example",
        model_name="spotify_recommendations",
        conda_env=conda_env,
        model_signature="auto",
    )

    return {
        "__default__": Pipeline([]),
        "prepare_mpd": prepare_mpd,
        "scrape_spotify_for_mpd": scrape_spotify_for_mpd,
        "insert_into_mongo": insert_into_mongo,
        "perform_cosine_experiment": perform_cosine_experiment,
        "mpd_cosine_ml": mpd_cosine_ml
    }
