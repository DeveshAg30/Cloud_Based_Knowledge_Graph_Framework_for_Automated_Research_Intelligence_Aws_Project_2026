"""openCypher clients for the three supported graph stores.

GRAPH_BACKEND selects one:
  * ``neptune-analytics`` (default) - Amazon Neptune Analytics graph, called through the AWS API
    (works from a Lambda outside a VPC when the graph has public connectivity + IAM auth)
  * ``neptune-db``  - Amazon Neptune Database cluster, openCypher HTTPS endpoint (Lambda must be in the VPC)
  * ``neo4j``       - Neo4j AuraDB Free (cloud, $0) or local Neo4j in Docker (docker-compose.yml)

All expose ``query(cypher, params) -> list[dict]``.
"""

import json
import os
import urllib.parse
import urllib.request


class NeptuneAnalyticsClient:
    def __init__(self, graph_id: str):
        import boto3

        self.graph_id = graph_id
        self.client = boto3.client("neptune-graph")

    def query(self, cypher: str, params: dict | None = None):
        kwargs = {"graphIdentifier": self.graph_id, "queryString": cypher, "language": "OPEN_CYPHER"}
        if params:
            kwargs["parameters"] = params
        resp = self.client.execute_query(**kwargs)
        return json.loads(resp["payload"].read()).get("results", [])


class NeptuneDBClient:
    def __init__(self, endpoint: str, port: int = 8182, iam_auth: bool = True):
        self.url = f"https://{endpoint}:{port}/openCypher"
        self.iam_auth = iam_auth

    def query(self, cypher: str, params: dict | None = None):
        body = {"query": cypher}
        if params:
            body["parameters"] = json.dumps(params)
        data = urllib.parse.urlencode(body).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if self.iam_auth:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            session = boto3.Session()
            req = AWSRequest(method="POST", url=self.url, data=data, headers=headers)
            SigV4Auth(session.get_credentials(), "neptune-db", session.region_name).add_auth(req)
            headers = dict(req.headers)
        request = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=60) as resp:
            return json.loads(resp.read()).get("results", [])


class Neo4jClient:
    """Local Docker Neo4j or Neo4j AuraDB Free (neo4j+s://xxxx.databases.neo4j.io)."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase

        from .schema import ALL_LABELS

        # Newer AuraDB instances name both the user and the database after the instance id.
        self.database = database or None
        try:  # schema notices ("relationship type does not exist yet") are expected and only add log noise
            self.driver = GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")
        except TypeError:  # older drivers without notification filtering
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._labels = sorted(ALL_LABELS)
        self._schema_ready = False

    def ensure_schema(self):
        """Neptune indexes everything automatically; Neo4j needs explicit indexes or MERGE becomes a full scan.
        Called by the writer only, so read-only API cold starts skip these 18 round trips."""
        if self._schema_ready:
            return
        for label in self._labels:
            # A uniqueness constraint (not a plain index): parallel ingest Lambdas MERGE the same entities, and without
            # it two concurrent MERGEs can both create the node. Graphs built before this change have a plain uid
            # index, which blocks the constraint, so it is dropped first. If the graph already holds duplicate uids
            # the constraint cannot be created; keep the plain index so ingestion still works until they are merged.
            uid_index = f"CREATE INDEX {label.lower()}_uid IF NOT EXISTS FOR (n:{label}) ON (n.uid)"
            try:
                self.query(f"DROP INDEX {label.lower()}_uid IF EXISTS")
                self.query(f"CREATE CONSTRAINT {label.lower()}_uid_unique IF NOT EXISTS "
                           f"FOR (n:{label}) REQUIRE n.uid IS UNIQUE")
            except Exception as exc:
                print(f"uniqueness constraint on {label}.uid not created ({exc}); using a plain index")
                self.query(uid_index)
            self.query(f"CREATE INDEX {label.lower()}_name IF NOT EXISTS FOR (n:{label}) ON (n.name_lc)")
        self._schema_ready = True

    def query(self, cypher: str, params: dict | None = None):
        with self.driver.session(database=self.database) as session:
            return [record.data() for record in session.run(cypher, params or {})]


_client = None


def get_graph_client():
    global _client
    if _client is not None:
        return _client
    backend = os.environ.get("GRAPH_BACKEND", "neptune-analytics")
    if backend == "neptune-analytics":
        _client = NeptuneAnalyticsClient(os.environ["NEPTUNE_GRAPH_ID"])
    elif backend == "neptune-db":
        _client = NeptuneDBClient(os.environ["NEPTUNE_ENDPOINT"], int(os.environ.get("NEPTUNE_PORT", "8182")),
                                  os.environ.get("NEPTUNE_IAM_AUTH", "true") == "true")
    elif backend == "neo4j":
        _client = Neo4jClient(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                              os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "research-kg"),
                              os.environ.get("NEO4J_DATABASE"))
    else:
        raise ValueError(f"Unknown GRAPH_BACKEND {backend!r}")
    return _client
