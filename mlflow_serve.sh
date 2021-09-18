#!/bin/bash

echo $MODEL_ID
echo $PORT

mlflow models serve -m runs:/$MODEL_ID/spotify_recommendations --no-conda --host 0.0.0.0 --port ${PORT:-5000}
