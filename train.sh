#!/bin/bash
# Usage: ./train.sh owner_id model_type use_sentiment use_cv
OWNER_ID=$1
MODEL_TYPE=$2
USE_SENTIMENT=$3
USE_CV=$4

python3 run_model_cli.py \
  --owner_id="$OWNER_ID" \
  --model_type="$MODEL_TYPE" \
  $( [ "$USE_SENTIMENT" = "true" ] && echo "--use_sentiment" ) \
  $( [ "$USE_CV" = "true" ] && echo "--cv" )
