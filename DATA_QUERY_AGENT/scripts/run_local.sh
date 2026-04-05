#!/usr/bin/env bash
set -euo pipefail

python -m llm_spark_pipeline \
  --question-file configs/question_example.txt \
  --metadata-file configs/metadata_example.json \
  --input-uri s3://your-bucket/input/ \
  --output-uri s3://your-bucket/output/result.xlsx

