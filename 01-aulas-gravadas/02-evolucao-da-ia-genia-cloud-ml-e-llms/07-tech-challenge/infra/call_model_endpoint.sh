awscurl --region sa-east-1 --service sagemaker \
  -X POST "https://runtime.sagemaker.sa-east-1.amazonaws.com/endpoints/avc-stroke-prediction-endpoint/invocations" \
  -H "Content-Type: application/json" \
  -d '[
  {
    "age": 53.0,
    "sbp": 118.0,
    "hba1c": 5.9,
    "bmi": 27.8,
    "gender": 0,
    "married": 1,
    "high_bp": 1,
    "chf": 1,
    "occupation": 0.0,
    "smoking": 1
  },
   {
    "age": 24.0,
    "sbp": 124.0,
    "hba1c": 5.6,
    "bmi": 40.8,
    "gender": 0,
    "married": 0,
    "high_bp": 0,
    "chf": 0,
    "occupation": 1.0,
    "smoking": 0
  }]'