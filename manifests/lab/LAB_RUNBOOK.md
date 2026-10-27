# Lab wiring — same pattern as customer OCP AI 3.5

Cluster: `api.ocp.lw97d.sandbox1049.opentlc.com`  
Project: `models-as-a-service`

## What we enabled / wired

| Layer | What | How |
|-------|------|-----|
| **OGX operator** | DSC `ogx: Managed` | Must set `llamastackoperator: Removed` first (deprecated conflict) |
| **MaaS LLM** | ExternalModel `granite-3-2-8b-instruct` → LiteLLM | Already present; OGX `remote::openai` uses that key/URL |
| **Embeddings** | OGX inline `granite-embedding-125m` | This lab LiteLLM has **no BGE**; customer with BGE MaaS → ExternalModel + remote embed |
| **pgvector** | `genai-pgvector` Deployment | Required by OGX RAG docs (you provision Postgres) |
| **OGXServer** | `lsd-genai-playground` | Files + Vector Stores + Responses |
| **App** | `quay.io/vraste/ogx-chatbot:1.2.0` | Env-only wiring to OGX / MLflow / EvalHub |
| **MLflow** | Operator CR Ready | `MLFLOW_TRACKING_URI` |
| **EvalHub** | Ready in `redhat-ods-applications` | Deep-link only |
| **Tempo** | DSCI `monitoring.traces` + OperatorHub Tempo/COO/OTel | Collector `…:4318` → Tempo |

## Exact steps we ran

### 1. Enable OGX (DSC)

```bash
# REQUIRED: LlamaStack must be Removed before OGX
oc patch datasciencecluster default-dsc --type=merge --patch-file=- <<'EOF'
{"spec":{"components":{"llamastackoperator":{"managementState":"Removed"},"ogx":{"managementState":"Managed"}}}}
EOF
oc get datasciencecluster default-dsc -o jsonpath='{.status.conditions[?(@.type=="OGXReady")]}'
# wait until OGXReady=True
```

### 2. MaaS external model (already on cluster)

```bash
oc get externalmodel,maasmodelref -n external-models
# granite-3-2-8b-instruct Ready via litellm-maas-rhdp
```

Copy LiteLLM API key into the demo project for OGX:

```bash
# secret maas-llm-api-key with label ogx.io/watch=true
```

### 3. Postgres + pgvector (OGX prerequisite)

```bash
oc apply -f manifests/lab/01-pgvector.yaml
# Use PGDATA subdirectory or emptyDir — do not mount PVC at PGDATA root
```

### 4. OGXServer

```bash
oc apply -f manifests/lab/02-ogxserver.yaml
oc get ogxserver -n models-as-a-service
# Phase=Ready, Service URL:
# http://lsd-genai-playground-service.models-as-a-service.svc.cluster.local:8321
```

Verify models:

```bash
curl http://lsd-genai-playground-service:8321/v1/models
# expects LLM maas-openai/granite-3-2-8b-instruct
# expects embedding sentence-transformers/ibm-granite/granite-embedding-125m-english
```

### 5. Deploy chatbot

```bash
oc apply -f manifests/lab/03-chatbot-app.yaml
oc get route ogx-chatbot -n models-as-a-service
```

### 6. Smoke test (verified on this lab)

```text
GET  /health  → ogx/mlflow/evalhub true; ingest via ogx_files_vector_stores
POST /ingest  → file_id + vector_store_id (OGX Files/Vector Stores)
POST /chat    → granite answer via OGX → MaaS
```

UI: `https://ogx-chatbot-models-as-a-service.apps.ocp.lw97d.sandbox1049.opentlc.com`

## Tempo / OpenTelemetry (why it was missing + how we fixed it)

**Why:** DSCI had `monitoring.managementState: Managed` but **no `traces` block**, so status was `Traces not configured in DSCI CR`. Also Tempo/COO/OpenTelemetry operators were not installed yet.

**Fix:**

```bash
# 1) OperatorHub (Red Hat): tempo-product, cluster-observability-operator, opentelemetry-product
# 2) Configure traces on DSCI:
oc patch dscinitialization default-dsci --type=merge -p '{
  "spec":{"monitoring":{"managementState":"Managed","traces":{
    "sampleRatio":"1.0",
    "storage":{"backend":"pv","size":"10Gi","retention":"168h"}
  }}}}'
# 3) Wait for collector:
oc get svc -n redhat-ods-monitoring | grep collector
# 4) Wire app Secret:
# OTEL_EXPORTER_OTLP_ENDPOINT=http://data-science-collector-collector.redhat-ods-monitoring.svc.cluster.local:4318
# OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

View traces: OpenShift AI → **Observe & monitor** → service `ogx-chatbot-api` (use `trace_id` from chat response).

## Customer differences to expect

1. **BGE**: register embedding ExternalModel / MaaS asset; set `OGX_EMBEDDING_MODEL` to that id; prefer remote embed over inline.
2. **MaaS gateway token**: mint dashboard API key into Secret (this lab used LiteLLM key after SA token got `403` on gateway).
3. **Dashboard path**: create OGX server from Gen AI playground UI instead of `overrideConfig` if preferred (same wiring: MaaS + pgvector + embed).

## Do not

- Install OGX/MLflow/EvalHub/Tempo via Helm outside DSC
- Point the chatbot at Postgres directly
- Leave `llamastackoperator: Managed` when enabling OGX
