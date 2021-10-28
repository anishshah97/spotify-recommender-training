# """
# Module for loading custom Cosine Similarity model using pandas as a pseduo db
# """
# import inspect
# import json
# import logging
# import os
# import pickle
# import tempfile
# from copy import deepcopy
# from distutils.dir_util import copy_tree
# # TODO: Rename as this can be used for general joblib/pickle loading
# from pathlib import Path

# import joblib
# import mlflow
# import pandas as pd
# import yaml
# from loguru import logger
# from mlflow import pyfunc
# from mlflow.exceptions import MlflowException
# from mlflow.models import Model, infer_signature
# from mlflow.models.model import MLMODEL_FILE_NAME
# from mlflow.models.signature import ModelSignature
# from mlflow.models.utils import ModelInputExample, _save_example
# from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
# from mlflow.tracking.artifact_utils import _download_artifact_from_uri
# from mlflow.utils.annotations import experimental
# from mlflow.utils.autologging_utils import (
#     ENSURE_AUTOLOGGING_ENABLED_TEXT,
#     INPUT_EXAMPLE_SAMPLE_ROWS,
#     InputExampleInfo,
#     MlflowAutologgingQueueingClient,
#     autologging_integration,
#     batch_metrics_logger,
#     exception_safe_function,
#     get_mlflow_run_params_for_fn_args,
#     resolve_input_example_and_signature,
#     safe_patch,
# )
# from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
# from mlflow.utils.environment import (
#     _CONDA_ENV_FILE_NAME,
#     _CONSTRAINTS_FILE_NAME,
#     _REQUIREMENTS_FILE_NAME,
#     _mlflow_conda_env,
#     _process_conda_env,
#     _process_pip_requirements,
#     _validate_env_arguments,
# )
# from mlflow.utils.file_utils import write_to
# from mlflow.utils.model_utils import _get_flavor_configuration
# from mlflow.utils.requirements_utils import _get_pinned_requirement

# from .models import CosineModel

# FLAVOR_NAME = "cosine_model"


# def get_default_pip_requirements():
#     """
#     :return: A list of default pip requirements for MLflow Models produced by this flavor.
#              Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
#              that, at minimum, contains these requirements.
#     """
#     # return [_get_pinned_requirement(FLAVOR_NAME)]
#     return None


# def get_default_conda_env():
#     """
#     :return: The default Conda environment for MLflow Models produced by calls to
#              :func:`save_model()` and :func:`log_model()`.
#     """
#     return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


# @format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
# def save_model(
#     cosine_model,
#     path,
#     conda_env=None,
#     mlflow_model=None,
#     signature: ModelSignature = None,
#     input_example: ModelInputExample = None,
#     pip_requirements=None,
#     extra_pip_requirements=None,
# ):
#     """
#     Save a CosineModel model to a path on the local file system.

#     :param cosine_model: Cosine model (an instance of `CosineModel`) to be saved.
#     :param path: Local path where the model is to be saved.
#     :param conda_env: {{ conda_env }}
#     :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

#     :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
#                       describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
#                       The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
#                       from datasets with valid model input (e.g. the training dataset with target
#                       column omitted) and valid model output (e.g. model predictions generated on
#                       the training dataset), for example:

#                       .. code-block:: python

#                         from mlflow.models.signature import infer_signature
#                         train = df.drop_column("target_label")
#                         predictions = ... # compute model predictions
#                         signature = infer_signature(train, predictions)
#     :param input_example: Input example provides one or several instances of valid
#                           model input. The example can be used as a hint of what data to feed the
#                           model. The given example will be converted to a Pandas DataFrame and then
#                           serialized to json using the Pandas split-oriented format. Bytes are
#                           base64-encoded.
#     :param pip_requirements: {{ pip_requirements }}
#     :param extra_pip_requirements: {{ extra_pip_requirements }}
#     """

#     _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

#     path = os.path.abspath(path)
#     if os.path.exists(path):
#         raise MlflowException("Path '{}' already exists".format(path))
#     model_data_subpath = "db.parquet"
#     model_data_path = os.path.join(path, model_data_subpath)
#     os.makedirs(path)
#     if mlflow_model is None:
#         mlflow_model = Model()
#     if signature is not None:
#         mlflow_model.signature = signature
#     if input_example is not None:
#         _save_example(mlflow_model, input_example, path)

#     # TODO: Move code copying functions to utils
#     # current_file_dir = Path(__file__).parent.resolve()
#     # # TODO: Remove hard coding in the loading scripts and pathing
#     # # BUG: Matching naming convention of current code base before injecting into mlproject
#     # new_code_dir = Path(path, "spotify_recommendations", "extras", "mlflow")
#     # copy_tree(str(current_file_dir), str(new_code_dir))
#     # for main_path, sub_paths, _ in os.walk(path):
#     #     for sub_path in sub_paths:
#     #         code_path = Path(main_path, sub_path)
#     #         init_file = Path(code_path, "__init__.py")
#     #         init_file.touch(exist_ok=True)
#     #         f = open(init_file)
#     #         f.close()

#     cosine_model.db.to_parquet(model_data_path)

#     pyfunc.add_to_model(
#         mlflow_model,
#         # code="spotify_recommendations",
#         loader_module="spotify_recommendations.extras.mlflow.loader_cosine_model",
#         data=model_data_subpath,
#         env=_CONDA_ENV_FILE_NAME,
#     )
#     mlflow_model.add_flavor(
#         FLAVOR_NAME, data=model_data_subpath)
#     mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

#     if conda_env is None:
#         if pip_requirements is None:
#             default_reqs = get_default_pip_requirements()
#             if not default_reqs:
#                 default_reqs = []
#             # To ensure `_load_pyfunc` can successfully load the model during the dependency
#             # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
#             inferred_reqs = mlflow.models.infer_pip_requirements(
#                 path, FLAVOR_NAME, fallback=default_reqs,
#             )
#             default_reqs = sorted(set(inferred_reqs).union(default_reqs))
#         else:
#             default_reqs = None
#         conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
#             default_reqs, pip_requirements, extra_pip_requirements,
#         )
#     else:
#         conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

#     with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
#         yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

#     # Save `constraints.txt` if necessary
#     if pip_constraints:
#         write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

#     # Save `requirements.txt`
#     write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


# @format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
# def log_model(
#     cosine_model,
#     artifact_path,
#     conda_env=None,
#     registered_model_name=None,
#     signature: ModelSignature = None,
#     input_example: ModelInputExample = None,
#     await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
#     pip_requirements=None,
#     extra_pip_requirements=None,
#     **kwargs
# ):
#     """
#     Log a CosineModel model as an MLflow artifact for the current run.

#     Save a CosineModel model to a path on the local file system.

#     :param cosine_model: Cosine model (an instance of `CosineModel`) to be saved.
#     :param artifact_path: Run-relative artifact path.
#     :param conda_env: {{ conda_env }}
#     :param registered_model_name: If given, create a model version under
#                                   ``registered_model_name``, also creating a registered model if one
#                                   with the given name does not exist.

#     :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
#                       describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
#                       The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
#                       from datasets with valid model input (e.g. the training dataset with target
#                       column omitted) and valid model output (e.g. model predictions generated on
#                       the training dataset), for example:

#                       .. code-block:: python

#                         from mlflow.models.signature import infer_signature
#                         train = df.drop_column("target_label")
#                         predictions = ... # compute model predictions
#                         signature = infer_signature(train, predictions)
#     :param input_example: Input example provides one or several instances of valid
#                           model input. The example can be used as a hint of what data to feed the
#                           model. The given example will be converted to a Pandas DataFrame and then
#                           serialized to json using the Pandas split-oriented format. Bytes are
#                           base64-encoded.
#     :param await_registration_for: Number of seconds to wait for the model version to finish
#                             being created and is in ``READY`` status. By default, the function
#                             waits for five minutes. Specify 0 or None to skip waiting.
#     :param pip_requirements: {{ pip_requirements }}
#     :param extra_pip_requirements: {{ extra_pip_requirements }}
#     :param kwargs: kwargs to pass to `lightgbm.Booster.save_model`_ method.
#     """
#     Model.log(
#         artifact_path=artifact_path,
#         flavor=mlflow.cosine_model,
#         registered_model_name=registered_model_name,
#         cosine_model=cosine_model,
#         conda_env=conda_env,
#         signature=signature,
#         input_example=input_example,
#         await_registration_for=await_registration_for,
#         pip_requirements=pip_requirements,
#         extra_pip_requirements=extra_pip_requirements,
#         **kwargs
#     )

# # BUG: Must reindex at track_spid


# def _load_model(path):
#     # Read from pickle or joblib
#     # model = pickle.load(open(path, "rb"))
#     # model = joblib.load(path)
#     model = CosineModel(pd.read_parquet(path).set_index("track_spid"))
#     return model


# def _load_pyfunc(path):
#     """
#     Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

#     """
#     return _load_model(path)


# def load_model(model_uri):
#     """
#     Load a LightGBM model from a local file or a run.

#     :param model_uri: The location, in URI format, of the MLflow model. For example:

#                       - ``/Users/me/path/to/local/model``
#                       - ``relative/path/to/local/model``
#                       - ``s3://my_bucket/path/to/model``
#                       - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

#                       For more information about supported URI schemes, see
#                       `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
#                       artifact-locations>`_.

#     :return: A LightGBM model (an instance of `lightgbm.Booster`_).
#     """
#     local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
#     flavor_conf = _get_flavor_configuration(
#         model_path=local_model_path, flavor_name=FLAVOR_NAME)
#     cosine_model_file_path = os.path.join(
#         local_model_path, flavor_conf.get("data", "db.parquet"))
#     return _load_model(path=cosine_model_file_path)
