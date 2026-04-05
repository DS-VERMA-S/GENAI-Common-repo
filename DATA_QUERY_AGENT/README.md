# LLM + Spark Action Planner (CLI)

This project turns a natural language question into a validated action plan, executes it on a Spark DataFrame, and writes the result to S3 (Excel by default).

**Flow**
1. User question + metadata -> LLM creates an action dictionary.
2. Validator checks the dictionary against metadata.
3. If validation fails, the LLM is re-prompted with errors.
4. Spark executes the validated plan and writes output to S3.

## Quick Start

1. Copy environment template:
```
cp .env.example .env
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the CLI:
```
python -m llm_spark_pipeline --question-file configs/question_example.txt --metadata-file configs/metadata_example.json --input-uri s3://your-bucket/input/ --output-uri s3://your-bucket/output/result.xlsx
```

## Notes

- Excel output is capped by `EXCEL_ROW_LIMIT` to avoid large collect operations.
- For S3 reads in local PySpark, you need Hadoop AWS jars and proper AWS credentials.

## Structure

- `src/llm_spark_pipeline`: core library
- `configs/`: sample metadata and question
- `scripts/`: helper scripts

