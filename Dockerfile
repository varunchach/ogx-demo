# OGX RAG chatbot — wires to OpenShift AI 3.5 cluster CRs only
# (OgxServer, MLflow, EvalHub, Tempo via data-science-collector).
# Build on OpenShift BuildConfig and push to Quay (see manifests/buildconfig.yaml).
FROM registry.access.redhat.com/ubi9/python-311:latest

LABEL name="ogx-chatbot" \
      version="1.4.20" \
      description="RAG demo via OGX Files/Vector Stores + MLflow + Tempo + EvalHub" \
      io.openshift.tags="rhoai,demo,ogx,rag,chatbot"

USER 0
WORKDIR /opt/app-root/src

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')"

COPY src/ src/
COPY config/ config/
COPY static/ static/
COPY data/ data/
COPY entrypoint.sh /opt/app-root/src/entrypoint.sh
RUN chmod +x /opt/app-root/src/entrypoint.sh && \
    chown -R 1001:0 /opt/app-root/src

USER 1001

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app-root/src

ENTRYPOINT ["/opt/app-root/src/entrypoint.sh"]
