# AWS implementation: step by step

> **On the AWS Free plan?** Neptune and Bedrock are blocked there. Follow [FREE_TIER_SETUP.md](FREE_TIER_SETUP.md) instead of Phases 1–2. It deploys the same stack with Neo4j AuraDB Free and the Gemini free tier, for $0. The other phases still apply.

These steps take the project from an empty AWS account to a working research-intelligence system. Commands are written for **PowerShell on Windows** and run from the project root. Region used throughout: **us-east-1**. It has Neptune Analytics, SageMaker serverless inference and Claude on Bedrock. Pick another region only if all three are available there.

| Phase | What you get | Owner |
|---|---|---|
| 0 | Tools, account safety, budget alarm | All |
| 1 | Bedrock access to Claude | Dhruv |
| 2 | Whole stack deployed: S3, Lambda, Neptune Analytics, API Gateway, Cognito, CloudWatch | Parth |
| 3 | Team user accounts | Parth |
| 4 | First papers in the graph (sample, OpenAlex, S2ORC) | Devesh |
| 5 | SciBERT NER trained on SageMaker and switched on | Devesh |
| 6 | Dashboard on Amplify | Parth |
| 7 | Monitoring and evaluation numbers for the report | All |
| 8 | Teardown, so you stop paying | All |

---

## Phase 0: Prerequisites

1. **AWS account.** Sign in as an IAM user or Identity Center user with admin rights, not as the root user.
2. **Budget alarm.** Do this first. Go to Billing → Budgets → Create budget → *Monthly cost budget*, set it to e.g. $25, and add an email alert at 80%. The Neptune Analytics graph bills every hour it exists, even when idle.
3. **Install tools:**
   ```powershell
   winget install Amazon.AWSCLI          # already installed: aws-cli 2.x
   winget install Amazon.SAM-CLI         # AWS SAM: builds and deploys template.yaml
   winget install Docker.DockerDesktop   # needed by `sam build --use-container`
   ```
   Start **Docker Desktop** and leave it running. Lambda runs on Linux, so its Python packages must be built inside a Linux container.
4. **Configure credentials and check them:**
   ```powershell
   aws configure            # access key, secret, region us-east-1, output json
   aws sts get-caller-identity
   ```
5. **Install the Python dependencies** for local scripts and tests:
   ```powershell
   pip install -r requirements-dev.txt
   python -m pytest tests
   ```

> **AWS Academy Learner Lab users:** Learner Lab blocks creating IAM roles and often blocks Neptune Analytics and Bedrock. Use a regular AWS account (free-tier sign-up plus the budget alarm above). If you must use Learner Lab, deploy with `GRAPH_BACKEND=neo4j` on an EC2 instance and `ExtractorMode=rules`.

## Phase 1: Enable Claude on Amazon Bedrock

1. There is nothing to "request" any more: the old **Model access** page is retired, and serverless models are enabled automatically the first time your account calls them. For **Anthropic** models only, a first-time account must submit use-case details once. Open **Amazon Bedrock** → **Model catalog** (us-east-1) → filter by provider **Anthropic** → open **Claude Opus 5.5**. If a banner asks for use-case details, fill it in (company: VIT Vellore; use case: academic research Q&A over scientific papers) and submit. Approval is usually immediate. If there's no banner, skip to step 2.
2. Smoke test from your machine (this first call is also what turns the model on for your account):
   ```powershell
   python -c "from anthropic import AnthropicBedrockMantle; c=AnthropicBedrockMantle(aws_region='us-east-1'); m=c.messages.create(model='anthropic.claude-opus-5-5', max_tokens=200, messages=[{'role':'user','content':'Say hello'}]); print(m.content[-1].text)"
   ```
   A `403` means the model access request has not gone through yet, or your user lacks `bedrock-mantle:CreateInference`.

The code calls Claude through the Bedrock Messages API endpoint (`AnthropicBedrockMantle`, model id `anthropic.claude-opus-5-5`). The Lambda roles in `template.yaml` are already granted `bedrock-mantle:CreateInference`.

## Phase 2: Deploy the stack with AWS SAM

```powershell
sam build --use-container
sam deploy --guided
```

Answers for the guided prompts:

| Prompt | Value |
|---|---|
| Stack Name | `research-kg` |
| AWS Region | `us-east-1` |
| Parameter Stage | `dev` |
| Parameter ExtractorMode | `llm` (zero-shot extraction with the configured LLM; `bedrock` is an accepted alias). Switch to `sagemaker` in Phase 5 |
| Parameter SageMakerEndpointName | `research-kg-ner` |
| Parameter BedrockModelId | `anthropic.claude-opus-5-5` |
| Parameter GraphMemory | `16` (smallest and cheapest) |
| Parameter DashboardUrl | `http://localhost:8080/` for now |
| Parameter AlarmEmail | your email, or leave empty |
| Confirm changes / allow IAM role creation / save config | `y` |
| "ApiFunction has no authentication" warning | Ignore. The Cognito JWT authorizer is the API's default authorizer |

The first deploy takes **15–25 minutes**, mostly spent creating the Neptune Analytics graph. When it finishes, print the outputs. You will need them in every later phase:

```powershell
aws cloudformation describe-stacks --stack-name research-kg --query "Stacks[0].Outputs" --output table
```

Save them as variables for this session:

```powershell
$out = aws cloudformation describe-stacks --stack-name research-kg --query "Stacks[0].Outputs" | ConvertFrom-Json
$BUCKET = ($out | Where-Object OutputKey -eq DataBucketName).OutputValue
$GRAPH  = ($out | Where-Object OutputKey -eq NeptuneGraphId).OutputValue
$POOL   = ($out | Where-Object OutputKey -eq UserPoolId).OutputValue
```

What the stack creates:

- **S3** `research-kg-<account>-us-east-1-dev`: encrypted, all public access blocked. Every object created under `raw/` triggers ingestion.
- **Neptune Analytics** graph `research-kg-dev`: public endpoint, so Lambda needs no VPC or NAT gateway. Every request still has to be IAM SigV4-signed.
- **Lambda** `research-kg-ingest-dev` (15 min timeout, with a dead-letter queue for failures) and `research-kg-api-dev`.
- **API Gateway HTTP API** with a Cognito JWT authorizer and CORS.
- **Cognito** user pool, app client (authorization code + PKCE) and hosted-UI domain.
- **CloudWatch** dashboard `research-kg-dev`, two alarms and an optional SNS email topic. X-Ray tracing is turned on for both functions.

After any code change, run `sam build --use-container; sam deploy`. The saved config is reused.

## Phase 3: Create the team's accounts

Self sign-up is disabled, so only invited researchers can log in:

```powershell
foreach ($email in "dhruv@example.com","parth@example.com","devesh@example.com") {
  aws cognito-idp admin-create-user --user-pool-id $POOL --username $email `
    --user-attributes Name=email,Value=$email Name=email_verified,Value=true
}
```

Each person gets an email with a temporary password and sets their own password at first login.

## Phase 4: Load research papers

### 4a. Sample data first, to check the pipeline end to end

```powershell
aws s3 cp dataset/sample_papers.jsonl "s3://$BUCKET/raw/sample/sample_papers.jsonl"
sam logs -n IngestFunction --stack-name research-kg --tail      # Ctrl+C once you see "ingested"
aws s3 ls "s3://$BUCKET/processed/"
```

Then check the graph itself:

```powershell
aws neptune-graph execute-query --graph-identifier $GRAPH --language OPEN_CYPHER `
  --query-string "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS c" out.json; Get-Content out.json
```

### 4b. OpenAlex: metadata, citations and open-access full text (recommended main source)

The full snapshot is about 330 GB, which is far more than this project needs. Pull a topical slice through the API instead. No key is needed for small runs: the free daily budget covers roughly 100 pages of 200 papers. For more, register a free key at openalex.org and set `$env:OPENALEX_API_KEY`.

```powershell
# Metadata + abstracts + citation links (fast, about 1 minute for 1000 papers)
python scripts/fetch_openalex.py --search "knowledge graph" --from-year 2019 --limit 1000 --email you@vit.ac.in

# Also download each paper's open-access PDF and extract the full text (about 4 s per paper)
python scripts/fetch_openalex.py --search "retrieval augmented generation" --limit 200 --with-fulltext --email you@vit.ac.in --out data/openalex_fulltext

aws s3 cp data/openalex "s3://$BUCKET/raw/openalex/" --recursive
aws s3 cp data/openalex_fulltext "s3://$BUCKET/raw/openalex_fulltext/" --recursive
```

What the flags do:
- `--min-references 1` (the default) skips papers with no reference list, because they would add no citation edges.
- Results are sorted by citation count unless you pass `--sort publication_date:desc`.
- Duplicate preprint and published versions of the same paper are dropped.
- **Full text without an API key:** only papers with an arXiv copy are selected. Publisher sites mostly return 403 to scripts, and PubMed Central serves a bot check.
- **Full text with an API key:** OpenAlex's own cached PDFs are also used.

Each output file holds 50 papers, so each Lambda invocation stays under its timeout and S3 processes the files in parallel.

### 4c. S2ORC: Semantic Scholar full-text corpus

The current S2ORC (`s2orc_v2`, about 16M parsed full-text papers) is distributed through the Semantic Scholar Datasets API:

1. Request a free API key at https://www.semanticscholar.org/product/api#api-key-form. It usually arrives by email within a few days.
2. Stream the shards and keep only the papers that match your keywords. Nothing multi-GB is saved to disk:
   ```powershell
   $env:S2_API_KEY="your-key"
   python scripts/fetch_s2orc.py --keywords "knowledge graph" "graph neural network" "relation extraction" --limit 300
   aws s3 cp data/s2orc "s3://$BUCKET/raw/s2orc/" --recursive
   ```
   The script resolves inline citations (`bib_ref` → `bib_entry` → cited paper's corpus id) and adds year, venue, abstract and author ids from the Graph API.
   - Each shard is 4–6 GB, so scanning is slow from a laptop. Running the script on a small EC2 instance in **us-west-2**, where the shards are stored, is much faster.
   - `--max-shards` and `--max-records` cap how much gets scanned.
3. Have the legacy 2020 S2ORC files (`metadata_N` + `pdf_parses_N`)? Use `scripts/prepare_s2orc.py` for those instead.

> The same paper fetched from both OpenAlex (`W…` id) and S2ORC (`s2-…` id) becomes two nodes. Load one source per topic, or use OpenAlex with `--with-fulltext`, which already gives metadata and full text together.

### 4d. PDFs

Upload PDFs from the dashboard (Phase 6), or directly with `aws s3 cp paper.pdf "s3://$BUCKET/raw/uploads/"`. Text is extracted with `pypdf`. Scanned PDFs have no text layer and need Amazon Textract, which is a possible extension.

**Large loads:** for tens of thousands of papers, run `scripts/local_ingest.py` from an EC2 instance, or export CSV and use Neptune Analytics' bulk `StartImportTask` from S3. The Lambda path is built for continuous, incremental ingestion.

## Phase 5: Train and deploy the SciBERT NER model on SageMaker

1. **Create the SageMaker execution role** (one time). Edit `src/aws/sagemaker-s3-policy.json` and replace `REPLACE_WITH_BUCKET` with `$BUCKET`'s value. Then:
   ```powershell
   aws iam create-role --role-name ResearchKG-SageMakerRole --assume-role-policy-document file://src/aws/sagemaker-trust.json
   aws iam attach-role-policy --role-name ResearchKG-SageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
   aws iam put-role-policy --role-name ResearchKG-SageMakerRole --policy-name data-bucket --policy-document file://src/aws/sagemaker-s3-policy.json
   $ROLE = aws iam get-role --role-name ResearchKG-SageMakerRole --query Role.Arn --output text
   ```
2. **Check your GPU quota.** New accounts often have a quota of 0 for training on `ml.g4dn.xlarge`. Go to Service Quotas → Amazon SageMaker → "ml.g4dn.xlarge for training job usage" and request 1. It is usually approved within a day. Without a GPU, pass `--train-instance ml.m5.2xlarge`. That works but takes several hours.
3. **Train and deploy.** The script downloads SciERC, uploads it to S3, fine-tunes `allenai/scibert_scivocab_uncased` for about 4 epochs (about 20 minutes on a g4dn), and deploys a **serverless** endpoint, which costs nothing while idle:
   ```powershell
   python scripts/sagemaker_launch.py --role $ROLE --bucket $BUCKET
   ```
   Precision, recall and F1 for dev and test are printed in the log. They are also stored as training-job metrics (SageMaker console → Training jobs → Metrics) and in `output/metrics.json`. Use these in your NLP performance report.
4. **Switch the pipeline to the trained model:**
   ```powershell
   sam deploy --parameter-overrides ExtractorMode=sagemaker
   ```
   Re-upload a file under `raw/` to re-ingest it. Upserts are idempotent, so this never creates duplicate nodes.

| ExtractorMode | When to use |
|---|---|
| `rules` | Offline or local testing. No cost, low recall |
| `bedrock` | Before the SageMaker model exists. Zero-shot extraction with Claude. Highest quality, highest cost per paper |
| `sagemaker` | The proposal's main path. Fine-tuned SciBERT. Cheapest at scale and produces measurable F1 numbers |

## Phase 6: Host the dashboard on AWS Amplify

1. Create `src/frontend/config.js` from `src/frontend/config.example.js` and fill in `apiUrl`, `cognitoDomain` and `clientId` from the stack outputs.
2. **Test locally:** run `python -m http.server 8080 --directory src/frontend` and open http://localhost:8080. Sign in, then ask a question.
3. **Deploy:**
   - Zip the *contents* of `src/frontend/` with `Compress-Archive src\frontend\* dashboard.zip -Force`.
   - Open the Amplify console → **Create new app** → **Deploy without Git** → upload `dashboard.zip`. You get a URL such as `https://main.d123abc.amplifyapp.com/`.
4. Point Cognito at the new URL and redeploy:
   ```powershell
   sam deploy --parameter-overrides DashboardUrl=https://main.d123abc.amplifyapp.com/
   ```
   In `config.js`, `redirectUri` is `window.location.origin + "/"`, so it works on both hosts. Cognito only allows the URL in `DashboardUrl`, though.

Dashboard features:
- natural-language question box with cited answers;
- a list of the retrieved graph facts, so you can check the answer was grounded;
- an interactive Cytoscape graph (click a node to expand it);
- graph statistics;
- drag-and-drop PDF/JSON upload, which goes to S3 through a presigned URL.

## Phase 7: Monitoring and evaluation

- **CloudWatch → Dashboards → `research-kg-dev`:**
  - documents processed, entities extracted and edges written;
  - per-document processing time;
  - query latency (average and p90);
  - Lambda errors.
- **Alarms:** the ingest Lambda errors alarm and the API errors alarm. They email you if you set `AlarmEmail`.
- **Logs Insights** (log group `/aws/lambda/research-kg-api-dev`). This shows what was asked and which entities were linked:
  ```
  fields @timestamp, question, seeds, facts, latency
  | filter ispresent(question) | sort @timestamp desc | limit 50
  ```
- **Failed uploads** land in the `IngestDeadLetterQueue` SQS queue. Inspect them, fix the cause, and re-upload.
- **X-Ray → Traces:** the time split between Neptune and Bedrock for each request.

Evaluation numbers for the report:
1. **NER quality:** SciERC test precision, recall and F1 from Phase 5.
2. **Processing speed (objective 2):** `DocumentLatency` and documents per hour, from the dashboard.
3. **GraphRAG grounding (objective 3):**
   - Write 30 questions whose answers you know from the ingested papers.
   - For each one, record answer correctness, the share of cited `doc:` ids that really support the claim (citation precision), and whether it refused when the graph lacked the answer.
   - Compare against asking Claude the same questions with no graph context. That baseline shows the hallucination reduction.
4. **Multi-hop discovery (objective 5):** show `PATH:` facts in answers, for example a method that links two papers that never cite each other.

## Phase 8: Costs and teardown

| Service | Cost driver | Notes |
|---|---|---|
| Neptune Analytics | Hourly, by provisioned memory, for as long as the graph exists | **Main cost.** At 16 m-NCU it is roughly $0.5/hour; check the Neptune pricing page for your region. Delete it when you are not working |
| Bedrock (Claude) | Tokens per query or extraction | Small for a class project. `ExtractorMode=llm` on thousands of papers adds up, so use `sagemaker` for bulk loads |
| SageMaker | Training job (per second) + serverless endpoint (per request) | The endpoint costs nothing while idle |
| Lambda, API Gateway, S3, Cognito, CloudWatch | Per use | Mostly within the free tier at project scale |

Tear everything down after the demo:

```powershell
aws s3 rm "s3://$BUCKET" --recursive          # buckets must be empty before deletion
sam delete --stack-name research-kg
aws sagemaker delete-endpoint --endpoint-name research-kg-ner
aws sagemaker delete-endpoint-config --endpoint-config-name research-kg-ner
# Amplify: delete the app in the console
```

To keep the graph data between work sessions, take a snapshot first: Neptune console → your graph → *Create snapshot*. Restore from it next time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `sam build` fails on `pydantic_core` / wheels | Use `sam build --use-container` with Docker Desktop running |
| Ingest Lambda `AccessDeniedException` on `neptune-graph` | The graph is still being created. Wait for the stack to reach `CREATE_COMPLETE` |
| `403` from Bedrock | Phase 1 model access is not approved yet, or you are in the wrong region |
| `/query` returns 503 after about 30 s | API Gateway has a hard 30 s limit. Keep `BEDROCK_EFFORT=low` (the default), or ask a more specific question so fewer facts are retrieved |
| Dashboard login loops back to the sign-in page | `DashboardUrl` must exactly match the page URL, including the trailing `/` |
| SageMaker `ResourceLimitExceeded` | Request the instance quota (Phase 5, step 2) or use a CPU instance |
| Neptune Database instead of Analytics | Set `GRAPH_BACKEND=neptune-db` and `NEPTUNE_ENDPOINT`. Put both Lambdas in the cluster's VPC, with VPC endpoints or a NAT gateway for Bedrock, SageMaker and S3 |
