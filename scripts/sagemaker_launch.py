"""Train the SciBERT NER model on SageMaker and deploy it as a serverless endpoint.

  python scripts/sagemaker_launch.py --role arn:aws:iam::<acct>:role/ResearchKG-SageMakerRole \
      --bucket <data-bucket> [--skip-train --model-data s3://.../model.tar.gz]

Requires: pip install "sagemaker>=2.230,<3"
"""

import argparse
import os
import tarfile
import tempfile
import urllib.request

import sagemaker
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker.serverless import ServerlessInferenceConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCIERC_URL = "http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
# Must be combinations the SageMaker SDK knows (sagemaker/image_uri_config/huggingface.json).
TRAINING_VERSIONS = {"transformers_version": "4.49.0", "pytorch_version": "2.5.1", "py_version": "py311"}
INFERENCE_VERSIONS = {"transformers_version": "4.49.0", "pytorch_version": "2.6.0", "py_version": "py312"}


def upload_scierc(session, bucket):
    work = tempfile.mkdtemp(prefix="scierc-")
    archive = os.path.join(work, "scierc.tar.gz")
    print(f"Downloading SciERC from {SCIERC_URL}")
    urllib.request.urlretrieve(SCIERC_URL, archive)
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(work, filter="data")
        except TypeError:  # Python < 3.10.12 has no extraction filters: reject unsafe paths ourselves
            root = os.path.realpath(work)
            for m in tar.getmembers():
                if not os.path.realpath(os.path.join(work, m.name)).startswith(root) or m.issym() or m.islnk():
                    raise RuntimeError(f"Unsafe path in archive: {m.name}")
            tar.extractall(work)
    json_dir = next(os.path.join(d, "") for d, _, files in os.walk(work) if "train.json" in files)
    return session.upload_data(json_dir, bucket=bucket, key_prefix="training/scierc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--endpoint-name", default="research-kg-ner")
    ap.add_argument("--train-instance", default="ml.g4dn.xlarge")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--model-data", help="existing model.tar.gz when --skip-train")
    args = ap.parse_args()

    session = sagemaker.Session()
    model_data = args.model_data
    if not args.skip_train:
        train_uri = upload_scierc(session, args.bucket)
        estimator = HuggingFace(
            entry_point="train.py",
            source_dir=os.path.join(ROOT, "src", "ai_model", "train"),
            role=args.role,
            instance_type=args.train_instance,
            instance_count=1,
            output_path=f"s3://{args.bucket}/models/",
            hyperparameters={"epochs": 4, "learning_rate": 3e-5, "batch_size": 16},
            metric_definitions=[
                {"Name": f"{split}:{m}", "Regex": f"{split}_{m}=([0-9.]+);"}
                for split in ("dev", "test") for m in ("precision", "recall", "f1")
            ],
            **TRAINING_VERSIONS,
        )
        estimator.fit({"train": train_uri})
        model_data = estimator.model_data
        print(f"Model artifact: {model_data}")

    model = HuggingFaceModel(
        model_data=model_data,
        role=args.role,
        entry_point="inference.py",
        source_dir=os.path.join(ROOT, "src", "ai_model", "inference"),
        **INFERENCE_VERSIONS,
    )
    model.deploy(
        endpoint_name=args.endpoint_name,
        # 3072 MB is the default per-endpoint serverless memory quota on new accounts
        serverless_inference_config=ServerlessInferenceConfig(memory_size_in_mb=3072, max_concurrency=10),
    )
    print(f"Endpoint ready: {args.endpoint_name}. Set SageMakerEndpointName={args.endpoint_name} "
          f"and ExtractorMode=sagemaker in the SAM deployment.")


if __name__ == "__main__":
    main()
