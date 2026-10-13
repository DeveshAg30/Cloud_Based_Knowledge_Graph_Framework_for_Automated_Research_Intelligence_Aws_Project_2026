"""Collect system performance numbers for the report from S3 and CloudWatch.

  python scripts/collect_metrics.py --days 7      # writes results/system_metrics.json

Read-only. Uses GetMetricStatistics (inside the CloudWatch free tier) and S3 reads of processed/*.json.
"""

import argparse
import datetime as dt
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

import boto3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(values, q):
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q * len(values)))], 2) if values else None


def ingestion_stats(bucket):
    s3 = boto3.client("s3")
    keys = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="processed/")
            for o in p.get("Contents", [])]

    def load(key):
        d = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        return d["stats"], d["document"].get("source"), len(" ".join(s["text"] for s in d["document"].get("sections", [])))

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(load, keys))
    secs = [r[0]["seconds"] for r in rows]
    full_text = [r for r in rows if r[2] > 5000]
    return {
        "documents": len(rows),
        "seconds_per_document": {"mean": round(statistics.mean(secs), 2), "p50": pct(secs, .5), "p90": pct(secs, .9)},
        "full_text_documents": len(full_text),
        "seconds_per_full_text_document_mean": round(statistics.mean(r[0]["seconds"] for r in full_text), 2) if full_text else None,
        "entities_per_document_mean": round(statistics.mean(r[0]["entities"] for r in rows), 1),
        "edges_per_document_mean": round(statistics.mean(r[0]["edges"] for r in rows), 1),
        "total_edges_written": sum(r[0]["edges"] for r in rows),
    }


def metric(cw, namespace, name, dims, stat, start, end, period=3600):
    resp = cw.get_metric_statistics(Namespace=namespace, MetricName=name, StartTime=start, EndTime=end, Period=period,
                                    Dimensions=[{"Name": k, "Value": v} for k, v in dims.items()],
                                    **({"ExtendedStatistics": [stat]} if stat.startswith("p") else {"Statistics": [stat]}))
    points = resp["Datapoints"]
    if not points:
        return None
    vals = [p["ExtendedStatistics"][stat] if stat.startswith("p") else p[stat] for p in points]
    if stat == "Sum" or stat == "SampleCount":
        return round(sum(vals), 2)
    if stat == "Maximum":
        return round(max(vals), 2)
    return round(statistics.mean(vals), 2)


def cloudwatch_stats(days, stage, endpoint):
    cw = boto3.client("cloudwatch")
    cfn = boto3.client("cloudformation")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in cfn.describe_stacks(StackName="research-kg")["Stacks"][0]["Outputs"]}
    api_id = outputs["ApiUrl"].split("//")[1].split(".")[0]
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days)
    sm = {"EndpointName": endpoint, "VariantName": "AllTraffic"}
    return {
        "window_days": days,
        "graphrag_query_latency_s": {"avg": metric(cw, "ResearchKG", "QueryLatency", {"Stage": "query"}, "Average", start, end),
                                     "p90": metric(cw, "ResearchKG", "QueryLatency", {"Stage": "query"}, "p90", start, end)},
        "questions_answered": metric(cw, "ResearchKG", "Queries", {"Stage": "query"}, "Sum", start, end),
        "documents_ingested": metric(cw, "ResearchKG", "DocumentsProcessed", {"Stage": "ingest"}, "Sum", start, end),
        "ingest_failures": metric(cw, "ResearchKG", "DocumentFailures", {"Stage": "ingest"}, "Sum", start, end) or 0,
        "api_requests": metric(cw, "AWS/ApiGateway", "Count", {"ApiId": api_id, "Stage": stage}, "Sum", start, end),
        "api_5xx": metric(cw, "AWS/ApiGateway", "5xx", {"ApiId": api_id, "Stage": stage}, "Sum", start, end) or 0,
        "api_latency_ms": {"avg": metric(cw, "AWS/ApiGateway", "Latency", {"ApiId": api_id, "Stage": stage}, "Average", start, end),
                           "p90": metric(cw, "AWS/ApiGateway", "Latency", {"ApiId": api_id, "Stage": stage}, "p90", start, end)},
        "sagemaker_invocations": metric(cw, "AWS/SageMaker", "Invocations", sm, "Sum", start, end),
        "sagemaker_model_latency_ms": (lambda v: round(v / 1000, 1) if v else None)(
            metric(cw, "AWS/SageMaker", "ModelLatency", sm, "Average", start, end)),
        "sagemaker_cold_starts": metric(cw, "AWS/SageMaker", "ModelSetupTime", sm, "SampleCount", start, end),
        "sagemaker_errors": (metric(cw, "AWS/SageMaker", "Invocation5XXErrors", sm, "Sum", start, end) or 0)
                            + (metric(cw, "AWS/SageMaker", "Invocation4XXErrors", sm, "Sum", start, end) or 0),
        "lambda_ingest_duration_ms_avg": metric(cw, "AWS/Lambda", "Duration", {"FunctionName": f"research-kg-ingest-{stage}"}, "Average", start, end),
        "lambda_api_duration_ms_avg": metric(cw, "AWS/Lambda", "Duration", {"FunctionName": f"research-kg-api-{stage}"}, "Average", start, end),
        "lambda_errors": (metric(cw, "AWS/Lambda", "Errors", {"FunctionName": f"research-kg-ingest-{stage}"}, "Sum", start, end) or 0)
                         + (metric(cw, "AWS/Lambda", "Errors", {"FunctionName": f"research-kg-api-{stage}"}, "Sum", start, end) or 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--stage", default="dev")
    ap.add_argument("--bucket", help="data bucket (default: DataBucketName output of the research-kg stack)")
    ap.add_argument("--endpoint", default="research-kg-ner")
    args = ap.parse_args()
    if not args.bucket:
        outputs = boto3.client("cloudformation").describe_stacks(StackName="research-kg")["Stacks"][0]["Outputs"]
        args.bucket = next(o["OutputValue"] for o in outputs if o["OutputKey"] == "DataBucketName")
    result = {"collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "ingestion": ingestion_stats(args.bucket),
              "cloudwatch": cloudwatch_stats(args.days, args.stage, args.endpoint)}
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump(result, open(os.path.join(ROOT, "results", "system_metrics.json"), "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
