# Running the project for $0 (AWS Free plan)

Our AWS account is on the **Free plan**: $100 of credits, and AWS **never charges the card**. The plan blocks **Amazon Neptune and Amazon Bedrock**, so day-to-day the project runs on free substitutes. Every other service is still AWS.

| Proposal component | Daily development (free) | Demo day (paid plan, paid from credits) |
|---|---|---|
| Graph database | **Neo4j AuraDB Free**: managed cloud graph, same openCypher queries | Amazon Neptune Analytics |
| LLM for GraphRAG | **Google Gemini API free tier** | Amazon Nova on Bedrock |
| NLP extraction | `rules` extractor (or Gemini, if it's within the free-tier limits) | same, or SageMaker SciBERT |
| S3, Lambda, API Gateway, Cognito, CloudWatch, Amplify | AWS | AWS |

To switch between the two columns you change a deploy parameter. The code is the same.

## 1. Create the free graph database (Neo4j AuraDB Free)

1. Sign up at https://console.neo4j.io. No card is needed.
2. Click **New instance** → **AuraDB Free** → **Create**.
3. **Download the credentials file** when it appears. The password is shown only once. The file contains:
   - `NEO4J_URI` (looks like `neo4j+s://xxxxxxxx.databases.neo4j.io`)
   - `NEO4J_USERNAME` (the instance id, e.g. `a1b2c3d4`; older instances use `neo4j`)
   - `NEO4J_PASSWORD`
   - `NEO4J_DATABASE` (also the instance id on newer instances)
   - Wait a few minutes after creating the instance before connecting.
4. Free tier limits:
   - 200,000 nodes and 400,000 relationships, which is plenty for a few thousand papers.
   - The instance **pauses after 3 days without use**. Click **Resume** in the console before working or demoing.

## 2. Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. No card is needed.
3. Free-tier limits are per minute and per day. They're fine for answering questions. For extracting hundreds of papers, keep `ExtractorMode=rules`.
4. On the free tier, Google may use prompts to improve its products. That's fine for public papers, but don't send private data.

## 3. Deploy

```powershell
cd C:\Users\dell\research-intelligence-kg
sam build --use-container
sam deploy --guided
```

Keep the earlier answers. The new prompts:

| Prompt | Value |
|---|---|
| GraphBackend | `neo4j` |
| Neo4jUri | the `NEO4J_URI` from the credentials file |
| Neo4jUser | `NEO4J_USERNAME` from the credentials file |
| Neo4jDatabase | `NEO4J_DATABASE` from the credentials file |
| Neo4jPassword | the `NEO4J_PASSWORD` (input is hidden) |
| LlmProvider | `gemini` |
| GeminiApiKey | your key (input is hidden) |
| GeminiModel | press Enter (`gemini-flash-lite-latest`: fast, and has capacity when bigger free models are overloaded) |
| ExtractorMode | `rules` |
| Disable rollback | `N` |

The deploy takes about 3 minutes; there's no Neptune graph to wait for. The secrets are saved in `samconfig.toml`, which is git-ignored, so **never commit it**. After that, Phases 3, 4 and 6 of [AWS_IMPLEMENTATION.md](AWS_IMPLEMENTATION.md) work unchanged: Cognito users, uploading papers, and the dashboard.

To run scripts locally against the same AuraDB:

```powershell
$env:GRAPH_BACKEND="neo4j"; $env:NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
$env:NEO4J_USER="xxxxxxxx"; $env:NEO4J_DATABASE="xxxxxxxx"; $env:NEO4J_PASSWORD="..."
$env:LLM_PROVIDER="gemini"; $env:GEMINI_API_KEY="..."
python scripts/local_ingest.py data/openalex --mode rules
python scripts/ask.py "Which methods are evaluated on WN18RR?"
```

To browse the graph, open the AuraDB console → **Query**, and run `MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100`.

## 3b. SciBERT NER without a SageMaker training quota (Phase 5)

On this account every SageMaker **training** instance quota is 0, but **serverless endpoints** are allowed (quota 5). So train for free on Google Colab's GPU, then host the trained model on SageMaker.

**In Google Colab** (https://colab.research.google.com):
1. Choose **Runtime → Change runtime type → T4 GPU**.
2. Click the folder icon and upload `src/ai_model/train/train.py`.
3. Run these cells:

```python
!pip -q install "transformers==4.46.3" datasets accelerate
```
```python
!wget -q http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz && tar xzf sciERC_processed.tar.gz
!find . -name train.json -path "*json*"      # note the folder it prints, e.g. ./processed_data/json
```
```python
!python train.py --data_dir processed_data/json --model_dir model --output_data_dir output --epochs 4
```
```python
!cat output/metrics.json                      # NER precision / recall / F1 (overall + per type): put these in the report
!tar czf model.tar.gz -C model .
from google.colab import files; files.download("model.tar.gz")
```

Training takes about 15–25 minutes on the free T4. The model file is about 420 MB.

**Or train on your own laptop's CPU, without Colab.** SciERC is small: 4 epochs took about 35 minutes on a 4-core i7.

```powershell
python -m venv trainenv
trainenv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
trainenv\Scripts\python -m pip install "transformers==4.46.3" datasets accelerate "sagemaker>=2.230,<3"
# download + extract sciERC_processed.tar.gz as above, then:
trainenv\Scripts\python sagemaker\train\train.py --data_dir processed_data\json --model_dir model --output_data_dir output --epochs 4
tar -czf model.tar.gz -C model .
```

**On your machine,** host the model on a SageMaker serverless endpoint. It's billed per request, and on the Free plan that comes out of credits:

```powershell
$BUCKET = aws cloudformation describe-stacks --stack-name research-kg --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" --output text
aws s3 cp $HOME\Downloads\model.tar.gz "s3://$BUCKET/models/scibert-ner/model.tar.gz"

# one-time SageMaker role (first replace REPLACE_WITH_BUCKET in src/aws/sagemaker-s3-policy.json)
aws iam create-role --role-name ResearchKG-SageMakerRole --assume-role-policy-document file://src/aws/sagemaker-trust.json
aws iam attach-role-policy --role-name ResearchKG-SageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam put-role-policy --role-name ResearchKG-SageMakerRole --policy-name data-bucket --policy-document file://src/aws/sagemaker-s3-policy.json
$ROLE = aws iam get-role --role-name ResearchKG-SageMakerRole --query Role.Arn --output text

pip install "sagemaker>=2.230,<3"
python scripts/sagemaker_launch.py --role $ROLE --bucket $BUCKET --skip-train --model-data "s3://$BUCKET/models/scibert-ner/model.tar.gz"
```

Then set `ExtractorMode=\"sagemaker\"` in `samconfig.toml`, run `sam deploy`, and re-ingest:
```powershell
aws s3 cp "s3://$BUCKET/raw/" "s3://$BUCKET/raw/" --recursive --metadata-directive REPLACE
```

If endpoint creation is refused on the Free plan, keep `ExtractorMode=rules`. The Colab F1 scores still cover the "fine-tuned NER" part of the report.

## 4. Demo day: show Neptune and Bedrock (optional)

Do this only if your instructor needs to see Neptune and Bedrock themselves. Read these first:
- **Upgrading is one-way.** After it, anything left running past your credits is charged to the card. AWS does not let you switch back to the Free plan.
- **Credits cover the demo.** Neptune Analytics at 16 m-NCU costs roughly $0.5/hour, and Nova questions cost fractions of a cent. A 4-hour window uses about $2–3 of the $100 credits.

Steps:

1. **Before upgrading,** create a zero-spend safety net: Billing → **Budgets** → *Create budget* → **Zero spend budget**, with your email. It warns you the moment anything costs real money beyond credits.
2. Billing → **Upgrade plan** → paid plan.
3. Deploy the Neptune and Nova configuration:
   ```powershell
   sam deploy --parameter-overrides GraphBackend=neptune-analytics LlmProvider=bedrock-nova ExtractorMode=rules AlarmEmail=you@example.com
   ```
   Wait about 20 minutes while the graph is created.
4. Re-ingest the papers already in S3 into Neptune. Copying the files onto themselves fires the upload trigger again:
   ```powershell
   aws s3 cp "s3://$BUCKET/raw/" "s3://$BUCKET/raw/" --recursive --metadata-directive REPLACE
   ```
5. Run the demo: dashboard questions and the CloudWatch dashboard.
6. **Immediately afterwards,** switch back. Switching back deletes the Neptune graph:
   ```powershell
   sam deploy
   aws neptune-graph list-graphs --query "graphs[].name"      # must print []
   ```

## What can still cost money

On the Free plan, nothing: AWS blocks the service instead of charging. The table only matters after upgrading.

| Resource | When it costs after upgrading | How to stop it |
|---|---|---|
| Neptune Analytics graph | Every hour it exists | `sam deploy` with `GraphBackend=neo4j`, or `sam delete` |
| SageMaker endpoint | Per request (serverless); training jobs per second | `aws sagemaker delete-endpoint --endpoint-name research-kg-ner` |
| Everything else in the stack | Within free-tier limits at project scale | `sam delete --stack-name research-kg` |

## Public sign-up and project members

Anyone can create an account on the dashboard (email + verification code) and use **Ask** and **Explore**.
**Uploading papers** is limited to the Cognito group `members`. To give someone upload rights:

```powershell
aws cognito-idp admin-add-user-to-group --user-pool-id <UserPoolId from the stack outputs> --username someone@example.com --group-name members
```

They need to sign out and back in for the change to take effect. The API is rate-limited (5 requests/s, burst 10) to protect
the Gemini free quota. Cognito's built-in email sender allows about 50 sign-up / reset emails per day.
