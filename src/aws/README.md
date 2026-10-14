# AWS infrastructure

The whole stack is one AWS SAM template: [`template.yaml`](../../template.yaml) at the repository root.
It covers S3, the two Lambdas, API Gateway (HTTP API, Cognito JWT, throttling), Cognito (user pool, `members` group), the optional
Neptune Analytics graph, SQS dead-letter queue, SNS, CloudWatch alarms, dashboard and saved Logs Insights queries, plus X-Ray.

This folder holds the IAM documents for the SageMaker execution role (`sagemaker-trust.json`, `sagemaker-s3-policy.json`).
Deployment steps: [`docs/FREE_TIER_SETUP.md`](../../docs/FREE_TIER_SETUP.md).
