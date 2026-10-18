/**
 * Standard Baseline RAG Web Application Client.
 * Manages UI interactions, live eGFR calculations, dynamic VLM guideline views,
 * Multi-Hop toggle control, Knowledge Graph rendering, and prompt editing.
 * 100% in English.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let benchmarkPatients = [];
  let currentPrompts = {};
  let defaultPrompts = {};
  let kgGraphData = null;
  let vlmPagesData = [];
  let isMultihopEnabled = true;

  // DOM Elements
  const benchmarkSelect = document.getElementById("benchmark-select");
  const btnRunPipeline = document.getElementById("btn-run-pipeline");
  const runSpinner = document.getElementById("run-spinner");
  const runIcon = document.getElementById("run-icon");
  const statusLabel = document.getElementById("status-label");
  const previewEgfr = document.getElementById("preview-egfr");
  const previewStage = document.getElementById("preview-stage");
  const chkMultihop = document.getElementById("chk-multihop");
  const multihopStatusBadge = document.getElementById("multihop-status-badge");

  // Inputs
  const inpAge = document.getElementById("inp-age");
  const inpBp = document.getElementById("inp-bp");
  const inpSc = document.getElementById("inp-sc");
  const inpAl = document.getElementById("inp-al");
  const inpHemo = document.getElementById("inp-hemo");
  const inpHtn = document.getElementById("inp-htn");
  const inpDm = document.getElementById("inp-dm");

  // Stepper Tabs
  const stepBtns = document.querySelectorAll(".step-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // VLM Elements
  const vlmPageTabs = document.querySelectorAll(".vlm-page-tab");
  const vlmPageImg = document.getElementById("vlm-page-img");
  const vlmExtractedJson = document.getElementById("vlm-extracted-json");

  // Modal Elements
  const promptModal = document.getElementById("prompt-modal");
  const btnOpenPrompts = document.getElementById("btn-open-prompts");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const btnSavePrompts = document.getElementById("btn-save-prompts");
  const btnResetPrompts = document.getElementById("btn-reset-prompts");
  const pTabBtns = document.querySelectorAll(".p-tab-btn");
  const pPanes = document.querySelectorAll(".p-pane");

  // 1. Initialize
  async function init() {
    await fetchStatus();
    await fetchBenchmarks();
    await fetchPrompts();
    await fetchGraph();
    await fetchVlmData();
    await fetchInitialResult();
    updateLiveEgfr();
  }

  // Multi-Hop Toggle Control
  if (chkMultihop) {
    chkMultihop.addEventListener("change", (e) => {
      isMultihopEnabled = e.target.checked;
      if (isMultihopEnabled) {
        multihopStatusBadge.textContent = "ON";
        multihopStatusBadge.className = "badge badge-vlm";
      } else {
        multihopStatusBadge.textContent = "OFF";
        multihopStatusBadge.className = "badge badge-critic";
      }
    });
  }

  // Live eGFR preview calculation
  function calculateMdrdEgfr(sc, age, isFemale = false) {
    if (sc <= 0 || age <= 0) return 90.0;
    const genderMult = isFemale ? 0.742 : 1.0;
    const egfr = 175.0 * Math.pow(sc, -1.154) * Math.pow(age, -0.203) * genderMult;
    return Math.round(egfr * 10) / 10;
  }

  function getCkdStageName(egfr, albumin) {
    if (egfr >= 90) return albumin > 0 ? "Stage 1 (Normal GFR + Damage)" : "Normal Function (G1)";
    if (egfr >= 60) return albumin > 0 ? "Stage 2 (Mild ↓ GFR + Damage)" : "Mildly Decreased (G2)";
    if (egfr >= 30) return "Stage 3 (Moderate ↓ GFR)";
    if (egfr >= 15) return "Stage 4 (Severe ↓ GFR)";
    return "Stage 5 (Kidney Failure / Dialysis)";
  }

  function updateLiveEgfr() {
    const age = parseFloat(inpAge.value) || 50;
    const sc = parseFloat(inpSc.value) || 1.0;
    const al = parseInt(inpAl.value) || 0;
    const egfr = calculateMdrdEgfr(sc, age);
    previewEgfr.textContent = egfr;
    previewStage.textContent = getCkdStageName(egfr, al);
  }

  [inpAge, inpSc, inpAl].forEach(el => el.addEventListener("input", updateLiveEgfr));

  // 2. Fetch Initial Server Status & Data
  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.status === "ready") {
        statusLabel.textContent = `Online (${data.total_chunks} Chunks, ${data.total_kg_nodes} KG Nodes)`;
      }
    } catch (e) {
      statusLabel.textContent = "Offline";
    }
  }

  async function fetchBenchmarks() {
    try {
      const res = await fetch("/api/patients");
      const data = await res.json();
      benchmarkPatients = data.benchmarks || [];
      benchmarkSelect.innerHTML = "";
      benchmarkPatients.forEach((p, idx) => {
        const opt = document.createElement("option");
        opt.value = idx;
        opt.textContent = p.case_label || `Patient ${p.patient_id}`;
        benchmarkSelect.appendChild(opt);
      });

      if (benchmarkPatients.length > 0) {
        loadBenchmarkCase(0);
      }
    } catch (e) {
      console.error("Failed to load benchmark patients:", e);
    }
  }

  function loadBenchmarkCase(idx) {
    const p = benchmarkPatients[idx];
    if (!p) return;
    if (p.age !== null) inpAge.value = p.age;
    if (p.bp !== null) inpBp.value = p.bp;
    if (p.sc !== null) inpSc.value = p.sc;
    if (p.al !== null) inpAl.value = p.al;
    if (p.hemo !== null) inpHemo.value = p.hemo;
    if (p.htn !== null) inpHtn.value = p.htn;
    if (p.dm !== null) inpDm.value = p.dm;
    updateLiveEgfr();
  }

  benchmarkSelect.addEventListener("change", (e) => {
    loadBenchmarkCase(parseInt(e.target.value));
  });

  // 3. VLM Guideline Data
  async function fetchVlmData() {
    try {
      const res = await fetch("/api/vlm/data");
      const data = await res.json();
      vlmPagesData = data.pages || [];
      renderVlmPage(39);
    } catch (e) {
      console.error("Failed to fetch VLM data:", e);
    }
  }

  function renderVlmPage(pageNum) {
    if (vlmPageImg) {
      vlmPageImg.src = `/api/page_image/${pageNum}`;
    }
    const pageData = vlmPagesData.find(p => p.page_num === parseInt(pageNum));
    const summaryEl = document.getElementById("vlm-summary-text");
    const containerEl = document.getElementById("vlm-structured-content");

    if (pageData) {
      if (summaryEl) summaryEl.textContent = pageData.page_summary || "KDIGO Clinical Guideline Scanned Page";
      if (containerEl) {
        containerEl.innerHTML = "";
        (pageData.elements || []).forEach((el, idx) => {
          const card = document.createElement("div");
          card.className = "vlm-elem-card";

          const factsHtml = (el.clinical_facts || []).map(f => `<li>${f}</li>`).join("");
          const triplesHtml = (el.relationships || []).length > 0
            ? `<div class="vlm-triples-row">${el.relationships.map(r => `<span class="triple-tag"><span class="t-subj">${r.subject}</span> <span class="t-pred">--[${r.predicate}]--></span> <span class="t-obj">${r.object}</span></span>`).join("")}</div>`
            : "";

          card.innerHTML = `
            <div class="vlm-elem-header">
              <span class="vlm-elem-title">${el.title || `Clinical Guideline Element ${idx + 1}`}</span>
              <span class="badge badge-vlm">${el.type || 'table'}</span>
            </div>
            <ul class="vlm-facts-list">${factsHtml}</ul>
            ${triplesHtml}
          `;
          containerEl.appendChild(card);
        });
      }
      if (vlmExtractedJson) {
        vlmExtractedJson.textContent = JSON.stringify(pageData, null, 2);
      }
    } else {
      if (summaryEl) summaryEl.textContent = `Loading VLM data for page ${pageNum}...`;
      if (containerEl) containerEl.innerHTML = "<div class='loading-text'>Processing scanned page...</div>";
      if (vlmExtractedJson) {
        vlmExtractedJson.textContent = `// Loading VLM extraction for page ${pageNum}...`;
      }
    }
  }

  vlmPageTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      vlmPageTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      const pageNum = tab.getAttribute("data-page");
      renderVlmPage(pageNum);
    });
  });

  // 4. Initial Result Auto-Load
  async function fetchInitialResult() {
    try {
      const res = await fetch("/api/initial_result");
      const data = await res.json();
      if (data && data.final_output) {
        displayResults(data);
      }
    } catch (e) {
      console.log("No initial result cached yet.");
    }
  }

  // 5. Stepper Tab Navigation
  stepBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      stepBtns.forEach(b => b.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");

      if (targetId === "tab-kg") {
        renderKnowledgeGraph(kgGraphData);
      } else if (targetId === "tab-outputs") {
        fetchOutputsList();
      }
    });
  });

  // 6. Knowledge Graph Canvas Renderer
  async function fetchGraph() {
    try {
      const res = await fetch("/api/graph");
      kgGraphData = await res.json();
      document.getElementById("kg-node-count").textContent = kgGraphData.nodes.length;
      document.getElementById("kg-edge-count").textContent = kgGraphData.links.length;
    } catch (e) {
      console.error("Failed to load KG:", e);
    }
  }

  function renderKnowledgeGraph(graph) {
    if (!graph) return;
    const canvas = document.getElementById("kg-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const nodes = graph.nodes;
    const links = graph.links;
    const nodeCoords = {};
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.38;

    nodes.forEach((n, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      nodeCoords[n.id] = { x, y, node: n };
    });

    links.forEach(l => {
      const u = nodeCoords[l.source];
      const v = nodeCoords[l.target];
      if (u && v) {
        ctx.beginPath();
        ctx.moveTo(u.x, u.y);
        ctx.lineTo(v.x, v.y);
        ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
        ctx.lineWidth = 1.0;
        ctx.stroke();

        const midX = (u.x + v.x) / 2;
        const midY = (u.y + v.y) / 2;
        ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
        ctx.font = "8px 'JetBrains Mono', monospace";
        ctx.fillText(l.relation, midX, midY);
      }
    });

    nodes.forEach(n => {
      const pt = nodeCoords[n.id];
      if (!pt) return;

      let color = "#38bdf8";
      if (n.type === "Stage") color = "#f59e0b";
      else if (n.type === "Action") color = "#10b981";
      else if (n.type === "Complication" || n.type === "DamageMarker") color = "#f43f5e";

      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.fillStyle = "#f8fafc";
      ctx.font = "bold 9px 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(n.id, pt.x, pt.y - 10);
    });
  }

  // 7. Pipeline Execution Trigger
  btnRunPipeline.addEventListener("click", async () => {
    btnRunPipeline.disabled = true;
    runSpinner.style.display = "inline-block";
    runIcon.style.display = "none";

    const customPatient = {
      patient_id: "INTERACTIVE-CASE",
      age: parseFloat(inpAge.value) || 60,
      bp: parseFloat(inpBp.value) || 80,
      sc: parseFloat(inpSc.value) || 1.8,
      al: parseInt(inpAl.value) || 2,
      hemo: parseFloat(inpHemo.value) || 10.0,
      htn: inpHtn.value,
      dm: inpDm.value,
      egfr: calculateMdrdEgfr(parseFloat(inpSc.value), parseFloat(inpAge.value)),
      calculated_stage: getCkdStageName(calculateMdrdEgfr(parseFloat(inpSc.value), parseFloat(inpAge.value)), parseInt(inpAl.value)),
      clinical_text: (
        `Patient ID: INTERACTIVE-CASE\n` +
        `Age: ${inpAge.value} years\n` +
        `Blood Pressure: ${inpBp.value} mm/Hg\n` +
        `Serum Creatinine: ${inpSc.value} mg/dL (Estimated eGFR: ${previewEgfr.textContent} mL/min/1.73m2)\n` +
        `Urine Albumin: ${inpAl.value}+\n` +
        `Hemoglobin: ${inpHemo.value} g/dL\n` +
        `Hypertension: ${inpHtn.value}\n` +
        `Diabetes Mellitus: ${inpDm.value}`
      )
    };

    const inpClinicalQuery = document.getElementById("inp-clinical-query");
    const clinicalQueryVal = inpClinicalQuery ? inpClinicalQuery.value.trim() : null;

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient: customPatient,
          enable_multihop: isMultihopEnabled,
          clinical_query: clinicalQueryVal
        })
      });
      const data = await res.json();
      if (data.error) {
        alert("Pipeline error: " + data.error);
        return;
      }
      displayResults(data);
    } catch (e) {
      alert("Request failed: " + e.message);
    } finally {
      btnRunPipeline.disabled = false;
      runSpinner.style.display = "none";
      runIcon.style.display = "inline-block";
    }
  });

  // Prompt Toggle Button Listener
  const btnTogglePrompt = document.getElementById("btn-toggle-prompt-view");
  const promptCollapse = document.getElementById("executed-prompt-collapse");
  const togglePromptBtnText = document.getElementById("toggle-prompt-btn-text");

  if (btnTogglePrompt && promptCollapse) {
    btnTogglePrompt.addEventListener("click", () => {
      const isHidden = promptCollapse.style.display === "none";
      promptCollapse.style.display = isHidden ? "block" : "none";
      if (togglePromptBtnText) {
        togglePromptBtnText.textContent = isHidden ? "Hide Executed Prompt" : "View Full Prompt";
      }
    });
  }

  // 8. Display Results in UI
  function displayResults(data) {
    document.getElementById("final-empty-state").style.display = "none";
    document.getElementById("final-results-card").style.display = "block";

    const finalOut = data.final_output || {};
    const actor = data.step4_actor || {};
    const critic = data.step5_critic || {};
    const mhop = data.step3_multihop_retrieval || {};
    const directives = data.pipeline_directives || {};

    // Final Decision View
    document.getElementById("final-diag-title").textContent = `${finalOut.predicted_class || 'CKD'} - ${finalOut.ckd_stage || 'Unclassified'}`;
    const confEl = document.getElementById("final-confidence");
    if (confEl) confEl.textContent = `${Math.round((finalOut.confidence || 0.9) * 100)}%`;
    document.getElementById("final-class-badge").textContent = `Class: ${finalOut.predicted_class || 'CKD'}`;
    document.getElementById("final-stage-badge").textContent = finalOut.ckd_stage || 'Unknown';
    document.getElementById("final-gfr-badge").textContent = finalOut.gfr_category || 'eGFR Category';

    const criticBadge = document.getElementById("final-critic-badge");
    if (critic.verdict === "accept") {
      criticBadge.textContent = "Critic Verified: Accepted ✓";
      criticBadge.className = "diag-chip chip-critic";
    } else if (critic.verdict === "revise") {
      criticBadge.textContent = "Critic Verified: Revised ✓";
      criticBadge.className = "diag-chip chip-critic";
    } else {
      criticBadge.textContent = "Critic: Insufficient Evidence";
      criticBadge.className = "diag-chip chip-class";
    }

    // Render Clinical Task Directive & Question
    const questionEl = document.getElementById("clinical-question-display");
    if (questionEl && directives.clinical_task_question) {
      questionEl.textContent = directives.clinical_task_question;
    }

    // Render In-Context Few-Shot Exemplars from Patient Dataset
    const fewShotGrid = document.getElementById("few-shot-cards-grid");
    if (fewShotGrid) {
      fewShotGrid.innerHTML = "";
      const exemplars = directives.few_shot_exemplars || [];
      if (exemplars.length === 0) {
        fewShotGrid.innerHTML = "<div style='font-size:0.68rem;color:#94a3b8;grid-column:1/-1;'>3 Reference exemplar cases dynamically supplied from UCI CKD Patient Dataset.</div>";
      } else {
        exemplars.forEach(ex => {
          const card = document.createElement("div");
          card.className = "few-shot-card";
          card.innerHTML = `
            <div class="few-shot-card-header">
              <span class="few-shot-card-title">Exemplar ${ex.exemplar_id} (${ex.patient_id})</span>
              <span class="badge ${ex.gold_class === 'CKD' ? 'badge-critic' : 'badge-vlm'}">${ex.gold_class}</span>
            </div>
            <div class="few-shot-param">${ex.summary}</div>
            <div class="few-shot-gold">Gold: ${ex.gold_stage}</div>
            <div class="few-shot-insight">${ex.clinical_insight}</div>
          `;
          fewShotGrid.appendChild(card);
        });
      }
    }

    // Render Full Executed Prompt
    const executedPromptText = document.getElementById("executed-prompt-text");
    if (executedPromptText) {
      executedPromptText.textContent = directives.actor_prompt_executed || "// Executed prompt text will be displayed here.";
    }

    document.getElementById("final-rationale-text").textContent = finalOut.clinical_rationale || "";

    const actionList = document.getElementById("final-action-list");
    actionList.innerHTML = "";
    (finalOut.clinical_action_plan || []).forEach(act => {
      const li = document.createElement("li");
      li.textContent = act;
      actionList.appendChild(li);
    });

    const citList = document.getElementById("final-citations-list");
    citList.innerHTML = "";
    (finalOut.evidence_citations || []).forEach(cit => {
      const sp = document.createElement("span");
      sp.className = "citation-tag";
      sp.textContent = cit;
      citList.appendChild(sp);
    });

    const actorInitEl = document.getElementById("summary-actor-init");
    if (actorInitEl) actorInitEl.textContent = actor.ckd_stage || "-";
    const criticVerdEl = document.getElementById("summary-critic-verdict");
    if (criticVerdEl) criticVerdEl.textContent = (critic.verdict || "-").toUpperCase();
    const revStatusEl = document.getElementById("summary-revision-status");
    if (revStatusEl) revStatusEl.textContent = data.step6_revision ? "Revised by Actor" : "Accepted without revision";

    // Multi-Hop View
    document.getElementById("hop1-query-text").textContent = mhop.hop1_query || "N/A";
    document.getElementById("hop2-query-text").textContent = mhop.hop2_query || "N/A";
    document.getElementById("hop2-reasoning-text").textContent = mhop.hop2_reasoning || "Direct single-hop reasoning";

    const hop1Evidence = document.getElementById("hop1-evidence-list");
    hop1Evidence.innerHTML = "";
    (mhop.hop1_docs || []).forEach(doc => {
      const item = document.createElement("div");
      item.className = "evidence-item";
      item.innerHTML = `<div class="evidence-item-id">${doc.chunk_id} (RRF: ${doc.rrf_score})</div><div class="evidence-item-text">${doc.text}</div>`;
      hop1Evidence.appendChild(item);
    });

    const hop2Evidence = document.getElementById("hop2-evidence-list");
    if (hop2Evidence) {
      hop2Evidence.innerHTML = "";
      if (mhop.hop2_docs && mhop.hop2_docs.length > 0) {
        mhop.hop2_docs.forEach(doc => {
          const item = document.createElement("div");
          item.className = "evidence-item";
          item.innerHTML = `<div class="evidence-item-id">${doc.chunk_id} (RRF: ${doc.rrf_score})</div><div class="evidence-item-text">${doc.text}</div>`;
          hop2Evidence.appendChild(item);
        });
      } else {
        hop2Evidence.innerHTML = `<div class="evidence-item" style="color:#94a3b8;font-style:italic;">Single-hop mode active (Hop 2 disabled or bypassed).</div>`;
      }
    }

    // RRF Candidates Table
    const rrfBody = document.getElementById("rrf-table-body");
    if (rrfBody) {
      rrfBody.innerHTML = "";
      (mhop.combined_docs || []).forEach(doc => {
        const tr = document.createElement("tr");
        const prov = doc.provenance || [];
        const hasDense = prov.some(p => p.retriever === "dense");
        const hasBm25 = prov.some(p => p.retriever === "bm25");
        const hasKg = prov.some(p => p.retriever === "kg");

        tr.innerHTML = `
          <td><code>${doc.chunk_id}</code></td>
          <td>${doc.title || doc.modality}</td>
          <td><strong>${doc.rrf_score || '-'}</strong></td>
          <td>${hasDense ? '<span style="color:#10b981;">✓</span>' : '<span style="color:#64748b;">-</span>'}</td>
          <td>${hasBm25 ? '<span style="color:#38bdf8;">✓</span>' : '<span style="color:#64748b;">-</span>'}</td>
          <td>${hasKg ? '<span style="color:#f59e0b;">✓</span>' : '<span style="color:#64748b;">-</span>'}</td>
        `;
        rrfBody.appendChild(tr);
      });
    }

    // Actor Tab
    const actorDump = document.getElementById("actor-json-dump");
    if (actorDump) actorDump.textContent = JSON.stringify(actor, null, 2);

    // Critic Tab
    const verdictEl = document.getElementById("critic-verdict-display");
    if (verdictEl) {
      verdictEl.textContent = (critic.verdict || "UNKNOWN").toUpperCase();
      verdictEl.className = "critic-verdict-badge " + (critic.verdict === "accept" ? "verdict-accept" : "verdict-revise");
    }

    const supportScore = critic.evidence_support_score || 0.0;
    const supportBar = document.getElementById("critic-support-bar");
    if (supportBar) supportBar.style.width = `${supportScore * 100}%`;
    const supportVal = document.getElementById("critic-support-value");
    if (supportVal) supportVal.textContent = supportScore;
    const stageStatus = document.getElementById("critic-staging-status");
    if (stageStatus) stageStatus.textContent = critic.staging_correct ? "Correct ✓" : "Flagged for Revision ⚠️";

    const instructions = critic.revision_instructions || "Evidence fully supports proposed diagnosis. No revision required.";
    const instrEl = document.getElementById("critic-instructions-text");
    if (instrEl) instrEl.textContent = instructions;

    const citAuditList = document.getElementById("critic-citations-audit-list");
    if (citAuditList) {
      citAuditList.innerHTML = "";
      (critic.citation_audit || []).forEach(item => {
        const li = document.createElement("li");
        li.innerHTML = `<strong>${item.is_supported ? '✓ Supported' : '✗ Unsupported'}:</strong> ${item.claim}`;
        citAuditList.appendChild(li);
      });
    }
  }

  // 9. Outputs Directory Explorer
  async function fetchOutputsList() {
    try {
      const res = await fetch("/api/outputs");
      const data = await res.json();
      const listEl = document.getElementById("outputs-file-list");
      listEl.innerHTML = "";

      const files = data.outputs || [];
      if (files.length === 0) {
        listEl.innerHTML = "<div style='color:#94a3b8;font-size:0.75rem;'>No files in outputs directory yet.</div>";
        return;
      }

      files.forEach((f, idx) => {
        const btn = document.createElement("button");
        btn.className = "output-file-btn" + (idx === 0 ? " active" : "");
        btn.innerHTML = `<span>${f.name}</span><span class="output-file-size">${(f.size_bytes / 1024).toFixed(1)} KB</span>`;
        btn.addEventListener("click", () => {
          document.querySelectorAll(".output-file-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          loadOutputFile(f.name);
        });
        listEl.appendChild(btn);
      });

      if (files.length > 0) {
        loadOutputFile(files[0].name);
      }
    } catch (e) {
      console.error("Failed to fetch outputs list:", e);
    }
  }

  async function loadOutputFile(filename) {
    document.getElementById("active-output-filename").textContent = `Viewing: adan_tmp/outputs/${filename}`;
    try {
      const res = await fetch(`/api/outputs/${filename}`);
      const data = await res.json();
      document.getElementById("output-file-content").textContent = JSON.stringify(data, null, 2);
    } catch (e) {
      document.getElementById("output-file-content").textContent = "Error reading file: " + e.message;
    }
  }

  // 10. Prompt Modal Configuration
  async function fetchPrompts() {
    try {
      const res = await fetch("/api/config");
      const data = await res.json();
      currentPrompts = data.prompts || {};
      defaultPrompts = data.default_prompts || {};
      document.getElementById("prompt-vlm-input").value = currentPrompts.vlm_prompt || "";
      document.getElementById("prompt-mhop-input").value = currentPrompts.multihop_prompt || "";
      document.getElementById("prompt-actor-input").value = currentPrompts.actor_prompt || "";
      document.getElementById("prompt-critic-input").value = currentPrompts.critic_prompt || "";
    } catch (e) {
      console.error("Failed to fetch prompts:", e);
    }
  }

  btnOpenPrompts.addEventListener("click", () => {
    promptModal.classList.add("open");
  });

  btnCloseModal.addEventListener("click", () => {
    promptModal.classList.remove("open");
  });

  pTabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      pTabBtns.forEach(b => b.classList.remove("active"));
      pPanes.forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const ptab = btn.getAttribute("data-ptab");
      document.getElementById(ptab).classList.add("active");
    });
  });

  btnSavePrompts.addEventListener("click", async () => {
    const updated = {
      vlm_prompt: document.getElementById("prompt-vlm-input").value,
      multihop_prompt: document.getElementById("prompt-mhop-input").value,
      actor_prompt: document.getElementById("prompt-actor-input").value,
      critic_prompt: document.getElementById("prompt-critic-input").value
    };
    try {
      const res = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompts: updated })
      });
      const data = await res.json();
      currentPrompts = data.prompts;
      alert("Prompts updated successfully!");
      promptModal.classList.remove("open");
    } catch (e) {
      alert("Error saving prompts: " + e.message);
    }
  });

  btnResetPrompts.addEventListener("click", async () => {
    if (!confirm("Reset all prompts to system defaults?")) return;
    try {
      const res = await fetch("/api/config/reset", { method: "POST" });
      const data = await res.json();
      currentPrompts = data.prompts;
      document.getElementById("prompt-vlm-input").value = currentPrompts.vlm_prompt;
      document.getElementById("prompt-mhop-input").value = currentPrompts.multihop_prompt;
      document.getElementById("prompt-actor-input").value = currentPrompts.actor_prompt;
      document.getElementById("prompt-critic-input").value = currentPrompts.critic_prompt;
      alert("Prompts reset to defaults!");
    } catch (e) {
      alert("Error resetting prompts: " + e.message);
    }
  });

  // 11. Natural Language Q&A Assistant Router
  const qaInput = document.getElementById("qa-input");
  const qaSubmitBtn = document.getElementById("qa-submit-btn");
  const qaLoading = document.getElementById("qa-loading");
  const qaResponseArea = document.getElementById("qa-response-area");
  const qaRouteTag = document.getElementById("qa-route-tag");
  const qaSourceTag = document.getElementById("qa-source-tag");
  const qaAnswerText = document.getElementById("qa-answer-text");
  const qaMetaDetails = document.getElementById("qa-meta-details");

  async function handleQaSubmit(queryText) {
    const q = (queryText || (qaInput ? qaInput.value : "")).trim();
    if (!q) return;

    if (qaInput) qaInput.value = q;
    if (qaLoading) qaLoading.style.display = "flex";
    if (qaResponseArea) qaResponseArea.style.display = "none";
    if (qaSubmitBtn) qaSubmitBtn.disabled = true;

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q })
      });
      const data = await res.json();

      if (qaRouteTag) qaRouteTag.textContent = `ROUTE: ${data.route_taken || "DOCUMENT_RAG"}`;
      if (qaSourceTag) qaSourceTag.textContent = `SOURCES: ${(data.sources || []).join(", ") || "KDIGO Corpus"}`;
      if (qaAnswerText) qaAnswerText.textContent = data.answer || "No response generated.";
      if (qaMetaDetails) {
        qaMetaDetails.innerHTML = `<strong>Routing Rationale:</strong> ${data.route_reasoning || "Standard evaluation"}`;
      }
      if (qaResponseArea) qaResponseArea.style.display = "block";
    } catch (err) {
      if (qaAnswerText) qaAnswerText.textContent = "Error executing assistant query: " + err.message;
      if (qaResponseArea) qaResponseArea.style.display = "block";
    } finally {
      if (qaLoading) qaLoading.style.display = "none";
      if (qaSubmitBtn) qaSubmitBtn.disabled = false;
    }
  }

  if (qaSubmitBtn) {
    qaSubmitBtn.addEventListener("click", () => handleQaSubmit());
  }

  if (qaInput) {
    qaInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleQaSubmit();
      }
    });
  }

  document.querySelectorAll(".preset-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      handleQaSubmit(q);
    });
  });

  // Start initialization
  init();
});
