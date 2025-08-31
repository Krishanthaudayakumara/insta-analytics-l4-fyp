#!/bin/bash
# Run model training for all available owner_ids
# Usage: ./train_all.sh [model_type] [use_sentiment] [use_cv]

MODEL_TYPE=${1:-xgboost}
USE_SENTIMENT=${2:-true}
USE_CV=${3:-true}

for f in outputs/high_value_followers_*.json; do
  OWNER_ID=$(basename "$f" | sed 's/high_value_followers_//;s/.json//')
  echo "=== Training for OWNER_ID: $OWNER_ID ==="
  ./train.sh "$OWNER_ID" "$MODEL_TYPE" "$USE_SENTIMENT" "$USE_CV"
done
