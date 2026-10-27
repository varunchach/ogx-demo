/* Bundled vanilla UI — no CDN */
(function () {
  const sessionId = "s-" + Math.random().toString(36).slice(2, 10);
  let config = null;
  let messages = [];
  let busy = false;
  let lastEvalPayload = null;

  const statusRow = document.getElementById("statusRow");
  const corpusMeta = document.getElementById("corpusMeta");
  const modelMeta = document.getElementById("modelMeta");
  const obsHint = document.getElementById("obsHint");
  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const fileEl = document.getElementById("file");
  const ingestInfoEl = document.getElementById("ingestInfo");
  const evalBtn = document.getElementById("evalBtn");
  const evalProvidersEl = document.getElementById("evalProviders");
  const evalPanel = document.getElementById("evalPanel");
  const evalClose = document.getElementById("evalClose");
  const evalRunBtn = document.getElementById("evalRunBtn");
  const evalResult = document.getElementById("evalResult");
  const evalPlatformHint = document.getElementById("evalPlatformHint");
  const errorEl = document.getElementById("error");

  function setError(msg) {
    errorEl.textContent = msg || "";
  }

  function statusChip(label, on) {
    const el = document.createElement("span");
    el.className = "status " + (on ? "on" : "off");
    el.innerHTML = "<i></i>";
    el.appendChild(document.createTextNode(label));
    return el;
  }

  function copyText(text, btn) {
    if (!text || !navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      const prev = btn.textContent;
      btn.textContent = "copied";
      setTimeout(function () { btn.textContent = prev; }, 1200);
    });
  }

  function pill(label, value) {
    const el = document.createElement("span");
    el.className = "pill";
    el.appendChild(document.createTextNode(label + " "));
    const code = document.createElement("span");
    code.textContent = value || "—";
    el.appendChild(code);
    if (value) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "copy";
      btn.addEventListener("click", function () { copyText(value, btn); });
      el.appendChild(btn);
    }
    return el;
  }

  function formatInlineMetrics(rm) {
    if (!rm || rm.skipped) return null;
    return (
      "Already scored on this chat turn (inline RAGAS-style):\n" +
      "faithfulness:          " + Number(rm.faithfulness).toFixed(3) + "\n" +
      "answer_relevancy:      " + Number(rm.answer_relevancy).toFixed(3) + "\n" +
      "context_sufficiency:   " + Number(rm.context_sufficiency).toFixed(3) + "\n" +
      "contexts used:         " + (rm.context_count != null ? rm.context_count : "—") + "\n" +
      "method:                " + (rm.method || "—") + "\n" +
      "notes:                 " + (rm.notes || "—") + "\n\n" +
      "Use Score last answer to re-run and log another MLflow eval run."
    );
  }

  function lastAssistantWithSources() {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === "assistant" && m.sources && m.sources.length && !m.guardrails_blocked) {
        return m;
      }
    }
    return null;
  }

  function openEvalPanel() {
    evalPanel.hidden = false;
    const last = lastAssistantWithSources();
    if (last && last.rag_metrics && !last.rag_metrics.skipped) {
      evalResult.textContent = formatInlineMetrics(last.rag_metrics);
      return;
    }
    if (!lastEvalPayload) {
      evalResult.textContent =
        "No RAG turn with retrieved sources yet. " +
        "Ask a document question that returns Sources, then try again.";
    }
  }

  function closeEvalPanel() {
    evalPanel.hidden = true;
  }

  async function loadConfig() {
    try {
      const r = await fetch("/config");
      config = await r.json();
      const flags = config.flags || {};
      statusRow.innerHTML = "";
      [
        ["OGX", flags.ogx],
        ["Vector", flags.vector_stores],
        ["MLflow", flags.mlflow],
        ["Tempo", flags.otel],
        ["NeMo", flags.guardrails],
        ["RAGAS", flags.rag_eval],
        ["EvalHub", flags.evalhub],
      ].forEach(function (pair) {
        statusRow.appendChild(statusChip(pair[0], !!pair[1]));
      });

      const files = config.corpus_files ?? 0;
      const store = config.vector_store_id || (config.ogx && config.ogx.vector_store_name) || "—";
      corpusMeta.textContent = files + " file(s) indexed · store " + store;

      const model = (config.ogx && config.ogx.model) || "—";
      const embed = (config.ogx && config.ogx.embedding_model) || "—";
      modelMeta.textContent = model + "\n" + embed;

      const obs = config.observability || {};
      obsHint.textContent =
        (obs.tempo_hint || "Observe → Traces · ogx-chatbot-api") +
        " · MLflow " + (obs.mlflow_experiment || "ogx-chatbot") +
        " / " + (obs.mlflow_workspace || "—");

      const eh = config.evalhub || config.evalhub_detail || {};
      const providers = eh.providers || (config.evalhub && config.evalhub.providers) || [];
      evalProvidersEl.textContent = providers.length
        ? "Platform providers: " + providers.join(" · ")
        : "In-app RAG scores → MLflow";
      evalPlatformHint.textContent =
        (eh.how_to_open_platform || eh.dashboard_hint ||
          "OpenShift AI → Evaluations for platform EvalHub batch jobs.") +
        (eh.base_url ? " (Route configured; not opened from this UI.)" : "");
      evalBtn.disabled = false;
      evalBtn.onclick = openEvalPanel;
    } catch (e) {
      setError("Cannot reach API /config");
    }
  }

  function renderMessages() {
    messagesEl.innerHTML = "";
    if (!messages.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.innerHTML =
        "<h2>Ask your documents</h2>" +
        "<p>Upload a PDF or text file, then ask a question. Retrieval runs through OGX Files and Vector Stores. Use <strong>Evaluations</strong> to score the last answer into MLflow (no golden document).</p>";
      messagesEl.appendChild(empty);
    }
    messages.forEach(function (m) {
      const wrap = document.createElement("div");
      wrap.className = "msg " + m.role;
      if (m.role === "assistant") {
        const role = document.createElement("div");
        role.className = "role";
        role.textContent = m.guardrails_blocked ? "Blocked by NeMo" : "Assistant";
        wrap.appendChild(role);
      }
      const bubble = document.createElement("div");
      bubble.className = "bubble" + (m.guardrails_blocked ? " blocked" : "");
      bubble.textContent = m.content;
      wrap.appendChild(bubble);

      if (m.role === "assistant" && (m.trace_id || m.mlflow_run_id || m.latency_ms || m.guardrails_blocked || m.rag_metrics)) {
        const row = document.createElement("div");
        row.className = "meta-row";
        if (m.guardrails_blocked) {
          const gr = document.createElement("span");
          gr.className = "pill warn";
          gr.textContent = "guardrails " + (m.guardrails_status || "blocked");
          row.appendChild(gr);
        }
        if (m.latency_ms != null) {
          const lat = document.createElement("span");
          lat.className = "pill";
          lat.textContent = Math.round(m.latency_ms) + " ms";
          row.appendChild(lat);
        }
        if (m.retrieve_hits != null) {
          const hits = document.createElement("span");
          hits.className = "pill";
          hits.textContent = "hits " + m.retrieve_hits;
          row.appendChild(hits);
        }
        const rm = m.rag_metrics;
        if (rm && !rm.skipped) {
          [
            ["faithfulness", rm.faithfulness],
            ["relevancy", rm.answer_relevancy],
            ["sufficiency", rm.context_sufficiency],
          ].forEach(function (pair) {
            if (pair[1] == null) return;
            const p = document.createElement("span");
            p.className = "pill metric";
            p.textContent = pair[0] + " " + Number(pair[1]).toFixed(2);
            row.appendChild(p);
          });
          if (rm.avg_retrieval_score != null) {
            const ars = document.createElement("span");
            ars.className = "pill";
            ars.textContent = "retrieve " + Number(rm.avg_retrieval_score).toFixed(2);
            row.appendChild(ars);
          }
        } else if (rm && rm.skipped) {
          const sk = document.createElement("span");
          sk.className = "pill";
          sk.textContent = "eval skipped";
          sk.title = rm.skip_reason || "";
          row.appendChild(sk);
        }
        if (m.trace_id) row.appendChild(pill("trace", m.trace_id));
        if (m.mlflow_run_id) row.appendChild(pill("mlflow", m.mlflow_run_id));
        wrap.appendChild(row);
        if (rm && rm.notes && !rm.skipped) {
          const note = document.createElement("div");
          note.className = "eval-notes";
          note.textContent = rm.notes;
          wrap.appendChild(note);
        }
      }

      if (m.sources && m.sources.length) {
        const details = document.createElement("details");
        details.className = "sources";
        const summary = document.createElement("summary");
        summary.textContent = "Sources · " + m.sources.length;
        details.appendChild(summary);
        m.sources.forEach(function (s, j) {
          const item = document.createElement("div");
          item.className = "source-item";
          const score = typeof s.score === "number" ? s.score.toFixed(3) : s.score;
          const name = (s.metadata && (s.metadata.filename || s.metadata.file_id)) || "chunk";
          const title = document.createElement("strong");
          title.textContent = (j + 1) + ". " + name + " · score " + score;
          item.appendChild(title);
          item.appendChild(document.createTextNode(s.content || ""));
          details.appendChild(item);
        });
        wrap.appendChild(details);
      }
      messagesEl.appendChild(wrap);
    });
    if (busy) {
      const thinking = document.createElement("div");
      thinking.className = "msg assistant";
      thinking.innerHTML = '<div class="bubble thinking">Working…</div>';
      messagesEl.appendChild(thinking);
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function runEvaluation() {
    // Prefer payload from last chat; rebuild from last assistant with sources if needed.
    let payload = lastEvalPayload;
    if (!payload) {
      const last = lastAssistantWithSources();
      if (last) {
        const q = (function () {
          for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === "user") return messages[i].content;
          }
          return "";
        })();
        payload = {
          question: q,
          answer: last.content,
          contexts: last.sources.map(function (s) { return s.content || ""; }),
          session_id: sessionId,
          trace_id: last.trace_id || "",
          chat_mlflow_run_id: last.mlflow_run_id || "",
        };
      }
    }
    if (!payload || !(payload.contexts && payload.contexts.length)) {
      evalResult.textContent =
        "No retrieved sources on the last answer. " +
        "file_search must return Sources before scoring.";
      return;
    }
    evalRunBtn.disabled = true;
    evalResult.textContent = "Scoring with OGX LLM judge (reference-free)…";
    try {
      const r = await fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      lastEvalPayload = payload;
      evalResult.textContent =
        "faithfulness:          " + j.faithfulness.toFixed(3) + "\n" +
        "answer_relevancy:      " + j.answer_relevancy.toFixed(3) + "\n" +
        "context_sufficiency:   " + j.context_sufficiency.toFixed(3) + "\n" +
        "contexts used:         " + j.context_count + "\n" +
        "method:                " + (j.method || "—") + "\n" +
        "mlflow_run_id:         " + (j.mlflow_run_id || "(not logged)") + "\n" +
        "notes:                 " + (j.notes || "—") + "\n\n" +
        "Logged to the same MLflow experiment as chat (kind=rag_evaluation).";
    } catch (e) {
      evalResult.textContent = "Evaluation failed: " + (e.message || e);
    } finally {
      evalRunBtn.disabled = false;
    }
  }

  async function send() {
    const text = inputEl.value.trim();
    if (!text || busy) return;
    setError("");
    inputEl.value = "";
    inputEl.style.height = "auto";
    const history = messages
      .filter(function (m) { return m.role === "user" || m.role === "assistant"; })
      .map(function (m) { return { role: m.role, content: m.content }; });
    messages.push({ role: "user", content: text });
    busy = true;
    renderMessages();
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          session_id: sessionId,
          history: history,
          use_rag: true,
        }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      const sources = j.sources || [];
      const blocked = !!j.guardrails_blocked;
      messages.push({
        role: "assistant",
        content: j.answer,
        trace_id: j.trace_id,
        mlflow_run_id: j.mlflow_run_id,
        sources: sources,
        latency_ms: j.latency_ms,
        retrieve_hits: j.retrieve_hits,
        guardrails_blocked: blocked,
        guardrails_status: j.guardrails_status || null,
        rag_metrics: j.rag_metrics || null,
      });
      // Keep payload whenever we have sources so Evaluations can re-score
      // even after inline RAGAS already ran on /chat.
      if (!blocked && sources.length) {
        lastEvalPayload = {
          question: text,
          answer: j.answer,
          contexts: sources.map(function (s) { return s.content || ""; }),
          session_id: sessionId,
          trace_id: j.trace_id || "",
          chat_mlflow_run_id: j.mlflow_run_id || "",
        };
      } else if (!blocked) {
        lastEvalPayload = null;
      }
    } catch (e) {
      setError(String(e.message || e));
      messages.push({ role: "assistant", content: "Error: " + (e.message || e) });
    } finally {
      busy = false;
      renderMessages();
    }
  }

  async function onUpload(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    busy = true;
    setError("");
    renderMessages();
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch("/ingest", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      ingestInfoEl.textContent =
        j.filename + " indexed · ~" + j.chunk_count + " chunks" +
        (j.deduped_removed ? " · replaced " + j.deduped_removed + " older" : "") +
        (j.trace_id ? " · trace " + j.trace_id.slice(0, 8) + "…" : "");
      await loadConfig();
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      busy = false;
      fileEl.value = "";
      renderMessages();
    }
  }

  sendBtn.addEventListener("click", send);
  fileEl.addEventListener("change", onUpload);
  evalClose.addEventListener("click", closeEvalPanel);
  evalRunBtn.addEventListener("click", runEvaluation);
  evalPanel.addEventListener("click", function (e) {
    if (e.target === evalPanel) closeEvalPanel();
  });
  inputEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  inputEl.addEventListener("input", function () {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  });

  loadConfig();
  renderMessages();
})();
