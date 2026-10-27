# OGX RAG Chatbot (OpenShift AI 3.5)

Source for the Quay image:

```text
quay.io/vraste/ogx-chatbot:1.4.20
```

This repository is the **exact build tree** used for that tag (`oc start-build --from-dir=.`).

| Component | How the app uses it |
|-----------|---------------------|
| **OgxServer** | `/v1/embeddings` (batched) · `vector_io.insert` / `vector_io.query` · chat/completions |
| **pgvector** | Owned by OGX — **not** opened by this app |
| **MLflow** | One run per ingest + chat turn |
| **Tempo** | OTLP → RHOAI `data-science-collector` (`ogx-chatbot-api`) |
| **EvalHub** | Deep-link + providers (batch/dashboard) |
| **NeMo Guardrails** | Optional `/v1/guardrail/checks` before OGX |

## Repository layout

```text
ogx-demo/
├── Dockerfile                 # Image build (tag 1.4.20)
├── requirements.txt
├── entrypoint.sh
├── .dockerignore
├── config/
│   └── settings.yaml          # Defaults (overridden by Secret/env)
├── src/                       # Application Python package
│   ├── api/                   # FastAPI routes (main.py, schemas.py)
│   ├── ogx_client.py          # OGX HTTP client (embed / vector_io / chat)
│   ├── ingest_chunking.py     # PDF/text extract + token-safe chunking
│   ├── rag.py                 # RAG façade
│   ├── rag_eval.py            # Inline LLM-as-judge metrics
│   ├── guardrails_client.py   # NeMo checks
│   ├── tracing.py             # OpenTelemetry → collector
│   ├── mlflow_logger.py
│   ├── evalhub_client.py
│   └── config.py
├── static/                    # Bundled UI (air-gapped)
│   ├── index.html
│   ├── app.js
│   └── app.css
├── data/                      # Sample corpus + golden Q&A
├── helm/ogx-chatbot/          # Helm chart (preferred deploy)
├── manifests/                 # Plain YAML + lab helpers
├── CLIENT_DEPLOY.md           # Customer deploy steps
└── README.md
```

## Ingest path (image 1.4.20)

1. Extract text (`pypdf` / plain text)
2. Chunk with `RecursiveCharacterTextSplitter` using the **BGE tokenizer** as `length_function` (hard cap &lt; 512 tokens)
3. `POST /v1/embeddings` in batches (default 16)
4. `vector_io.insert` into the configured vector store  
Chat retrieve: `vector_io.query` + chat/completions (file_search fallback).

## Strict platform-only rule

Enable OGX / MLflow / EvalHub / Tempo via **dashboard / DSC / Operators** first.  
This folder deploys only the chatbot app (Deployment + Service + Route + Secret).

## Helm install

```bash
cp helm/ogx-chatbot/values-customer.example.yaml values-customer.yaml
# edit values-customer.yaml
helm upgrade --install ogx-chatbot ./helm/ogx-chatbot \
  -n <demo-ns> -f values-customer.yaml --create-namespace
```

## Plain YAML deploy

```bash
oc apply -f manifests/deploy-oneshot.yaml
oc get route ogx-chatbot -n <demo-ns>
```

## Rebuild image (must match this repo)

```bash
oc apply -f manifests/buildconfig.yaml -n <demo-ns>
oc start-build ogx-chatbot --from-dir=. --follow -n <demo-ns>
```

## Local run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OGX_BASE_URL=... OGX_API_KEY=... OGX_MODEL=... OGX_EMBEDDING_MODEL=...
export PYTHONPATH=.
uvicorn src.api.main:app --reload --port 8000
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Bundled UI |
| GET | `/health` | Wiring flags |
| GET | `/config` | Wiring detail |
| GET | `/readyz` | Liveness/readiness |
| POST | `/ingest` | multipart upload → chunk/embed/insert |
| POST | `/chat` | RAG chat |

Full customer steps: [CLIENT_DEPLOY.md](CLIENT_DEPLOY.md).
