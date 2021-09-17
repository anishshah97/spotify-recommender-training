#!/bin/bash

echo $MODEL_ID

mlflow models serve -m runs:/$MODEL_ID/spotify_recommendations --no-conda --host 0.0.0.0 --port 5000
