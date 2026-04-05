param(
  [string]$QuestionFile = "configs/question_example.txt",
  [string]$MetadataFile = "configs/metadata_example.json",
  [string]$InputUri = "s3://your-bucket/input/",
  [string]$OutputUri = "s3://your-bucket/output/result.xlsx"
)

python -m llm_spark_pipeline `
  --question-file $QuestionFile `
  --metadata-file $MetadataFile `
  --input-uri $InputUri `
  --output-uri $OutputUri

