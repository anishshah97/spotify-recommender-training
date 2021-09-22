# Spotify Recommendations

## Overview

### The Problem @ ✋

- I like Spotify and I like discovering music
- Music generation is very popularity based nowadays
- Discovery/interaction with music happens via playlists mostly nowadays
- Music curation tools for playlists are scarce
- Spotify has a lot of tools for interacting with their data/features

![](./static/imgs/discover-weekly.png)

### Million Playlist Dataset
![](./static/imgs/mpd-logo.png)

- Spotify hosted the 2018 RecSys challenge of 1 Million Playlists made by users
- Publicly released the dataset
- No associated metadata of tracks

![](./static/imgs/mpd-overview.png)
![](./static/imgs/mpd-features-json.png)

Track Metadata - Spotify Web API
![](./static/imgs/spotify-web-auth-flow.png)
![](./static/imgs/spotify-audio-features-json.png)

EDA - MPD + Track Metadata
![](./static/imgs/danceability-eda-example.png)
![](./static/imgs/eda-most-featured-artists-mpd.png)

### The Approach 🕵️

- Build infrastructure/tooling to easily build and deploy Spotify-based applications and models
- Do so in a streamlined reproducible fashion
- Automate as much as possible
- Infrastructure-as-code as much as possible
- Make it easy for a SWE/DE/DS/MLE/DevOps person to extend upon
- Use a MLFlow wrapped custom cosine similarity model as a test of E2E tooling from the ML end

#### Infrastructure considerations

- arbitrary custom model can be made easily
- models can connect to growing data sources
- deployment of said model is straightforward
- can be managed by ci/cd easily
- adaptable to data, models, and code changes

#### The Experiment - Let’s Keep it Simple

- Data: Create training and test data by way of obfuscation on MPD
- Take all the playlists and randomly remove a certain amount of songs and use as training data
- Use those removed songs and use as testing data
- Task: Use training songs in the playlist to predict the songs that were removed
- Method: Aggregate features of the songs to create a representative feature representation for the playlist w.r.t songs and use a similarity search
- Features used: Spotify given track features
- Search: Cosine Similarity

#### Introducing SpotifyPlaylists! 🎶

High Level Software Architecture
![](./static/imgs/basic-app-example.png)
![](./static/imgs/simple-etl-serving.png)
![](./static/imgs/advanced-etl-serving.png)

Deployment Stack
![](./static/imgs/cdk-visualized.png)
![](./static/imgs/cloudformation-visualized.png)
![](./static/imgs/my-cloudformation-stacks.png)

CodePipeline Stack
![](./static/imgs/codepipeline-visualized.png)
![](./static/imgs/my-cloudformation-codepipeline-stack.png)
![](./static/imgs/my-cloudformation-codepipeline-paramters.png)
![](./static/imgs/my-codepipeline.png)

Pre-requirements: EKS Cluster (using eksctl)
![](./static/imgs/my-cluster.png)

MLFlow Stack
![](./static/imgs/mlflow-cdk.png)
![](./static/imgs/mlflow-cdk-output.png)
![](./static/imgs/my-mlflow-cdk-stack.png)
![](./static/imgs/my-mlflow-cdk-outputs)
![](./static/imgs/mlflow-dashboard-example)

Codebase - Kedro
![](./static/imgs/kedro-architecture.png)
![](./static/imgs/data-engineering-data-layers.png)
![](./static/imgs/kedro-node-reuse-visualized.png)

An Aside - DAGS
![](./static/imgs/kubeflow-dag-example.png)
![](./static/imgs/airflow-dag-example.png)

Kedro - MLFlow
![](./static/imgs/kedro-mflow-use-cases-visualized.png)

Training
![](./static/imgs/my-kedro-etl.png)
![](./static/imgs/my-kedro-training.png)

The Dynamic Duo in Action
![](./static/imgs/my-deployed-mlflow.png)
![](./static/imgs/my-mlflow-model-artifact.png)
![](./static/imgs/my-mlflow-model-registry.png)

Pipelines as Models??? - Inference Served
![](./static/imgs/my-kedro-inference.png)

You’re sure that’s the model?
![](./static/imgs/mlflow-models-registry-serving-use-case.png)

Demo App

- MLFlow: http://deplo-mlflo-16g0oyp6k65hv-5cd5faf094caf332.elb.us-east-1.amazonaws.com/#/
- Model Endpoint:
  https://spotify-recommendations-crun-ctreuw63uq-uw.a.run.app/invocations

Demo Video of Endpoint

Recap
![](./static/imgs/e2e-cicd-etl-mlops-pipeline.png)

#TODO:

- Add Prometheus and Grafana metrics
- Replace with an actually good model
- Connect all components of my application (Front-end and Back-end)
- Use MLFlow Model Registry to add streamlined Blue/Green Deployments and Canary Deployments
- Use CDK to spin up the CloudFormation template to spin up the aforementioned CodePipeline Stack
- Fix codebase to package minimal requirements for a better served model container
- Port from custom containers to something like a Seldon deployment if possible

## Development

### Kedro Overview

This is your new Kedro project, which was generated using `Kedro 0.17.4`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

### Rules and guidelines

In order to get the best out of the template:

- Don't remove any lines from the `.gitignore` file we provide
- Make sure your results can be reproduced by following a [data engineering convention](https://kedro.readthedocs.io/en/stable/12_faq/01_faq.html#what-is-data-engineering-convention)
- Don't commit data to your repository
- Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

### How to install dependencies

Declare any dependencies in `pyproject.toml` as per usage with `poetry`. You can also use the `src/requirements.txt` is you use the `kedro install` and `build-reqs` flow.

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
kedro test
```

To configure the coverage threshold, go to the `.coveragerc` file.

### Project dependencies

For the `poetry` workflow, to generate/update dependencies, use the `poetry add` functionality alongside adjusting the `pyproject.toml` to take advantages of `poetry`'s automatic dependancy resolution

If you are using the `kedro install` workflow, to generate or update the dependency requirements for your project: `kedro build-reqs`

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.

[Further information about project dependencies](https://kedro.readthedocs.io/en/stable/04_kedro_project_setup/01_dependencies.html#project-specific-dependencies)

### How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `kedro install` you will not need to take any extra steps before you use them.

#### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

#### JupyterLab

To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

#### IPython

And if you want to run an IPython session:

```
kedro ipython
```

#### How to convert notebook cells to nodes in a Kedro project

You can move notebook code over into a Kedro project structure using a mixture of [cell tagging](https://jupyter-notebook.readthedocs.io/en/stable/changelog.html#release-5-0-0) and Kedro CLI commands.

By adding the `node` tag to a cell and running the command below, the cell's source code will be copied over to a Python file within `src/<package_name>/nodes/`:

```
kedro jupyter convert <filepath_to_my_notebook>
```

> _Note:_ The name of the Python file matches the name of the original notebook.

Alternatively, you may want to transform all your notebooks in one go. Run the following command to convert all notebook files found in the project root directory and under any of its sub-folders:

```
kedro jupyter convert --all
```

#### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can run `kedro activate-nbstripout`. This will add a hook in `.git/config` which will run `nbstripout` before anything is committed to `git`.

> _Note:_ Your output cells will be retained locally.

### Package your Kedro project

[Further information about building project documentation and packaging your project](https://kedro.readthedocs.io/en/stable/03_tutorial/05_package_a_project.html)
