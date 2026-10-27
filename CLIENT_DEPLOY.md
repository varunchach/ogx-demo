# Client deploy — OGX RAG Chatbot (OpenShift AI 3.5)

## What this zip contains

| Path | Purpose |
|------|---------|
| `helm/ogx-chatbot/` | **Preferred** Helm chart (`helm install`) |
| `manifests/deploy-oneshot.yaml` | Plain YAML one-shot (same resources) |
| `manifests/buildconfig.yaml` | Optional: build image on cluster → Quay |
| `src/`, `static/`, `Dockerfile` | Source (rebuild if you cannot pull Quay) |
| `data/` | Sample doc + golden Q&A for demos |

**Image (pre-built):** `quay.io/vraste/ogx-chatbot:1.3.0`

This package deploys **only the chatbot app**. Platform pieces must already exist on the cluster.

## Prerequisites (platform — dashboard / DSC)

Do these **before** install:

1. OpenShift AI: OGX (Managed), MLflow, TrustyAI/EvalHub, monitoring traces (Tempo + collector)
2. MaaS: LLM + embedding model available to OgxServer
3. An **OgxServer** (and pgvector provider) in the demo project
4. Project labeled for MLflow workspace: `opendatahub.io/dashboard=true`
5. Pull secret `quay-creds` in the demo namespace (or change image to an internal registry)

## Deploy with Helm (recommended)

```bash
cd ogx-chatbot

# 1) Customer values
cp helm/ogx-chatbot/values-customer.example.yaml values-customer.yaml
# Edit: namespace, OGX URL, models, MLflow workspace, EVALHUB_BASE_URL

# 2) Pull secret (once)
oc create secret docker-registry quay-creds \
  --docker-server=quay.io \
  --docker-username='<user>' \
  --docker-password='<token>' \
  --docker-email=unused@example.com \
  -n <demo-ns> --dry-run=client -o yaml | oc apply -f -

# 3) Install / upgrade
helm upgrade --install ogx-chatbot ./helm/ogx-chatbot \
  -n <demo-ns> \
  -f values-customer.yaml \
  --create-namespace

# 4) UI URL
oc get route ogx-chatbot -n <demo-ns> -o jsonpath='https://{.spec.host}{"\n"}'
```

Cluster-scoped Tempo RBAC is included (`tempoRbac.enabled: true`). Needs cluster-admin (or equivalent) for those ClusterRole(s). Set `tempoRbac.enabled: false` if platform already grants collector write.

## Deploy with plain YAML (alternative)

1. Edit `manifests/deploy-oneshot.yaml` → Secret `stringData` + namespaces  
2. Pull secret (same as above)  
3. Apply:

```bash
oc apply -f manifests/deploy-oneshot.yaml
oc get route ogx-chatbot -n <demo-ns> -o jsonpath='https://{.spec.host}{"\n"}'
```

## After deploy

Open the Route → upload a PDF → ask a question.

- Tempo: Observe → Traces · `{ name = "POST /chat" }` · service `ogx-chatbot-api`  
- MLflow: OpenShift AI → Experiments (MLflow) · experiment `ogx-chatbot`

## Optional: rebuild image on customer cluster

```bash
oc apply -f manifests/buildconfig.yaml -n <demo-ns>
# Edit BuildConfig output image if needed
oc start-build ogx-chatbot --from-dir=. --follow -n <demo-ns>
```

## EvalHub URL

```bash
oc get route -A | findstr evalhub
# or
oc get route evalhub -n <evalhub-namespace> -o jsonpath='https://{.spec.host}{"\n"}'
```

## Support note

Do **not** point the app at Postgres/pgvector directly. Chunk/embed/store stay behind OgxServer.
