#!/bin/bash

echo $RUN_ID

mlflow models serve -m runs:/$RUN_ID/spotify_recommendations --no-conda --host 0.0.0.0 --port 5000
