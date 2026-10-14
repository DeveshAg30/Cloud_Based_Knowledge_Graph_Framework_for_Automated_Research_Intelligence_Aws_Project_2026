"""CloudWatch custom metrics via Embedded Metric Format: printing this JSON to stdout in Lambda
creates metrics in the ResearchKG namespace without any PutMetricData calls."""

import json
import time

NAMESPACE = "ResearchKG"


def emit(metrics: dict, unit: str = "Count", **dimensions):
    print(json.dumps({
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [list(dimensions)] if dimensions else [[]],
                "Metrics": [{"Name": k, "Unit": unit} for k in metrics],
            }],
        },
        **dimensions,
        **metrics,
    }))
