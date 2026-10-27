# OGX RAG Chatbot (OpenShift AI 3.5)

Minimal **RAG** demo that wires an app to **cluster-installed** RHOAI components.

| Component | How the app uses it |
|-----------|---------------------|
| **OgxServer** | Files + Vector Stores (ingest) · Responses `file_search` (RAG) · chat fallback |
| **pgvector** | Owned by OGX — **not** opened by this app |
| **MLflow** | Tracking URI → one run per ingest + chat turn |
| **Tempo** | OTLP → RHOAI `data-science-collector` |
| **EvalHub** | Deep-link + providers (CR in `redhat-ods-applications`) |

## Strict platform-only rule

Enable OGX / MLflow / EvalHub / Tempo via **dashboard / DSC / Operators** first.  
This folder deploys only Deployment + Service + Route + Secret.

**Do not** install platform components, connect to Postgres directly, or load UI from public CDNs.

## Image

```
quay.io/vraste/ogx-chatbot:1.3.0
```

Bundled static UI (`static/index.html` + `app.js` + `app.css`) — works air-gapped.

**Client handoff:** see [CLIENT_DEPLOY.md](CLIENT_DEPLOY.md).

## Helm install (recommended)

```bash
cp helm/ogx-chatbot/values-customer.example.yaml values-customer.yaml
# edit values-customer.yaml
helm upgrade --install ogx-chatbot ./helm/ogx-chatbot \
  -n <demo-ns> -f values-customer.yaml --create-namespace
```

## Plain YAML deploy

Edit [`manifests/deploy-oneshot.yaml`](manifests/deploy-oneshot.yaml), then:

```bash
oc apply -f manifests/deploy-oneshot.yaml
oc get route ogx-chatbot -n <demo-ns>
```

Required Secret keys (no `PGVECTOR_*`):

```text
OGX_BASE_URL, OGX_API_KEY, OGX_MODEL, OGX_EMBEDDING_MODEL
OGX_EMBEDDING_DIMENSION, OGX_VECTOR_STORE_NAME, OGX_VECTOR_PROVIDER_ID
OTEL_EXPORTER_OTLP_ENDPOINT, MLFLOW_TRACKING_URI, EVALHUB_BASE_URL
```

Optional: `OGX_VECTOR_STORE_ID` to pin an existing store.

## Build & push to Quay

```bash
cd ogx-chatbot
oc apply -f manifests/buildconfig.yaml -n <demo-ns>
oc start-build ogx-chatbot --from-dir=. --follow -n <demo-ns>
```

Make `quay.io/vraste/ogx-chatbot:1.2.0` public; link `quay-creds` for pull.

## Customer demo checklist

1. Open Route → confirm wiring chips (OGX, VectorStore, BGE, MLflow, Tempo, EvalHub).
2. Upload `data/sample_policy.txt` → OGX Files + Vector Store attach; note `trace_id` / `mlflow_run_id`.
3. Ask a question → Responses `file_search`; note `trace_id` → Tempo (`ogx-chatbot-api`).
4. MLflow experiment `ogx-chatbot` → ingest + chat runs.
5. **Open EvalHub** → run providers from dashboard using `data/golden_qa.json`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Bundled UI |
| GET | `/health` | Wiring flags (no secrets) |
| GET | `/config` | Wiring detail |
| GET | `/evalhub` | Providers + probe |
| GET | `/ogx/models` | Proxy OGX models |
| POST | `/ingest` | multipart → OGX Files + Vector Store |
| POST | `/chat` | RAG via OGX Responses / chat |

## Local run

```bash
cd ogx-chatbot
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OGX_BASE_URL=... OGX_API_KEY=... OGX_MODEL=... OGX_EMBEDDING_MODEL=...
export PYTHONPATH=.
uvicorn src.api.main:app --reload --port 8000
```
