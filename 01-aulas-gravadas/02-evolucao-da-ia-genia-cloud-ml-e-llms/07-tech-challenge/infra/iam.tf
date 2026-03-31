# ==============================================================================
# IAM Role para SageMaker Notebook e Training Jobs
# ==============================================================================

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${local.name_prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Política para acesso ao S3 (bucket do projeto)
resource "aws_iam_role_policy" "sagemaker_s3_policy" {
  name = "${local.name_prefix}-s3-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "s3:GetBucketAcl",
          "s3:PutObjectAcl"
        ]
        Resource = [
          aws_s3_bucket.data_bucket.arn,
          "${aws_s3_bucket.data_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Política para SageMaker (criar training jobs, endpoints, etc.)
resource "aws_iam_role_policy" "sagemaker_full_policy" {
  name = "${local.name_prefix}-sagemaker-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:InvokeEndpoint",
          "sagemaker:ListTags",
          "sagemaker:AddTags",
          "sagemaker:CreateAutoMLJob",
          "sagemaker:DescribeAutoMLJob",
          "sagemaker:ListAutoMLJobs",
          "sagemaker:ListCandidatesForAutoMLJob",
          "sagemaker:CreateHyperParameterTuningJob",
          "sagemaker:DescribeHyperParameterTuningJob",
          "sagemaker:StopHyperParameterTuningJob",
          "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
          "sagemaker:CreateExperiment",
          "sagemaker:DescribeExperiment",
          "sagemaker:DeleteExperiment",
          "sagemaker:CreateTrial",
          "sagemaker:DescribeTrial",
          "sagemaker:DeleteTrial",
          "sagemaker:CreateTrialComponent",
          "sagemaker:DescribeTrialComponent",
          "sagemaker:DeleteTrialComponent",
          "sagemaker:UpdateTrialComponent",
          "sagemaker:AssociateTrialComponent",
          "sagemaker:DisassociateTrialComponent",
          "sagemaker:BatchPutMetrics",
          "sagemaker:CreateFeatureGroup",
          "sagemaker:DescribeFeatureGroup",
          "sagemaker:DeleteFeatureGroup",
          "sagemaker:PutRecord",
          "sagemaker:GetRecord",
          "sagemaker:BatchGetRecord",
          "sagemaker:CreatePipeline",
          "sagemaker:DescribePipeline",
          "sagemaker:UpdatePipeline",
          "sagemaker:DeletePipeline",
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:StopPipelineExecution",
          "sagemaker:ListPipelineExecutionSteps"
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:*"
      },
      {
        Sid    = "SageMakerGlobalActions"
        Effect = "Allow"
        Action = [
          "sagemaker:Search",
          "sagemaker:ListExperiments",
          "sagemaker:ListTrials",
          "sagemaker:ListTrialComponents",
          "sagemaker:ListTrainingJobs",
          "sagemaker:ListModels",
          "sagemaker:ListEndpoints",
          "sagemaker:ListFeatureGroups",
          "sagemaker:ListPipelines",
          "sagemaker:ListPipelineExecutions"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/sagemaker/*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_execution_role.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "iam:GetRole"
        ]
        Resource = aws_iam_role.sagemaker_execution_role.arn
      },
      {
        Effect = "Allow"
        Action = [
          "glue:CreateDatabase",
          "glue:GetDatabase",
          "glue:CreateTable",
          "glue:GetTable",
          "glue:UpdateTable",
          "glue:DeleteTable",
          "glue:GetPartitions",
          "glue:BatchCreatePartition"
        ]
        Resource = [
          "arn:aws:glue:${var.aws_region}:${local.account_id}:catalog",
          "arn:aws:glue:${var.aws_region}:${local.account_id}:database/sagemaker_featurestore",
          "arn:aws:glue:${var.aws_region}:${local.account_id}:table/sagemaker_featurestore/*"
        ]
      }
    ]
  })
}

# Política para CloudWatch Metrics (monitoramento do endpoint)
resource "aws_iam_role_policy" "sagemaker_cloudwatch_policy" {
  name = "${local.name_prefix}-cloudwatch-access"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })
}
