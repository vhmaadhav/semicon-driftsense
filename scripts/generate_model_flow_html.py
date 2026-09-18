#!/usr/bin/env python3
"""Generate self-contained interactive visualization HTML for DriftSense model flow.

Combines the extracted feature traces from feature_traces.json into an interactive,
responsive HTML viewer with dark/light themes, stage scrubbers, case switchers,
and detailed mathematical explanations.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
TRACES_PATH = os.path.join(PROJECT_ROOT, "docs", "model_visualizer", "feature_traces.json")
OUTPUT_HTML_PROJECT = os.path.join(PROJECT_ROOT, "docs", "model_visualizer", "index.html")
OUTPUT_HTML_ROOT = os.path.join(PROJECT_ROOT, "..", "semicon-driftsense-model-flow.html")

TEMPLATE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DriftSenseNet — Visual Model Feature Flow</title>
<style>
  :root {
    --bg: #0d1117;
    --card-bg: #161b22;
    --card-border: #30363d;
    --text: #c9d1d9;
    --text-bright: #f0f6fc;
    --text-dim: #8b949e;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --cyan: #38bdf8;
    --emerald: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --code-bg: #090d13;
  }

  [data-theme="light"] {
    --bg: #f6f8fa;
    --card-bg: #ffffff;
    --card-border: #d0d7de;
    --text: #24292f;
    --text-bright: #0969da;
    --text-dim: #57606a;
    --accent: #6d28d9;
    --accent-light: #7c3aed;
    --cyan: #0284c7;
    --emerald: #059669;
    --amber: #d97706;
    --rose: #e11d48;
    --code-bg: #f6f8fa;
  }

  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }

  header {
    background: var(--card-bg);
    border-bottom: 1px solid var(--card-border);
    padding: 20px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
  }

  header h1 {
    margin: 0 0 4px 0;
    font-size: 22px;
    color: var(--text-bright);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .badge {
    font-size: 12px;
    background: var(--accent);
    color: #fff;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 600;
  }

  .header-actions {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .btn {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    color: var(--text);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    text-decoration: none;
    transition: all 0.15s ease;
  }

  .btn:hover {
    border-color: var(--accent-light);
    color: var(--text-bright);
  }

  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .btn-primary:hover {
    background: var(--accent-light);
    border-color: var(--accent-light);
  }

  .container {
    max-width: 1440px;
    margin: 0 auto;
    padding: 24px 32px;
  }

  /* Case Selector Tabs */
  .case-tabs {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }

  .case-tab {
    flex: 1;
    min-width: 300px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.1s;
    position: relative;
  }

  .case-tab:hover {
    border-color: var(--cyan);
  }

  .case-tab.active {
    border-color: var(--cyan);
    box-shadow: 0 0 0 2px var(--cyan);
  }

  .case-tab h3 {
    margin: 0 0 6px 0;
    font-size: 15px;
    color: var(--text-bright);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .case-tab p {
    margin: 0;
    font-size: 13px;
    color: var(--text-dim);
  }

  .status-pill {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
  }
  .status-pill.pass { background: rgba(52, 211, 153, 0.2); color: var(--emerald); border: 1px solid var(--emerald); }
  .status-pill.reject { background: rgba(251, 113, 133, 0.2); color: var(--rose); border: 1px solid var(--rose); }

  /* Stage Stepper */
  .stepper {
    display: flex;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 8px;
    margin-bottom: 24px;
    overflow-x: auto;
    gap: 6px;
  }

  .step-btn {
    flex: 1;
    min-width: 140px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 10px 8px;
    color: var(--text-dim);
    cursor: pointer;
    text-align: center;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.15s ease;
  }

  .step-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-bright);
  }

  .step-btn.active {
    background: var(--accent);
    color: #ffffff;
    font-weight: 600;
  }

  .step-btn span.num {
    display: block;
    font-size: 11px;
    opacity: 0.8;
    margin-bottom: 2px;
  }

  /* Main Workspace Layout */
  .workspace-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 24px;
  }

  @media (max-width: 1080px) {
    .workspace-grid {
      grid-template-columns: 1fr;
    }
  }

  .viewer-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 10px;
    padding: 24px;
  }

  .viewer-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 14px;
  }

  .viewer-title h2 {
    margin: 0 0 6px 0;
    font-size: 18px;
    color: var(--text-bright);
  }

  .viewer-title p {
    margin: 0;
    font-size: 13px;
    color: var(--text-dim);
  }

  .stage-media-container {
    display: flex;
    gap: 20px;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    background: var(--code-bg);
    border-radius: 8px;
    padding: 24px;
    min-height: 480px;
    border: 1px solid var(--card-border);
    position: relative;
  }

  .image-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    position: relative;
  }

  .image-panel img {
    max-width: 440px;
    max-height: 440px;
    border-radius: 6px;
    border: 1px solid var(--card-border);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    image-rendering: auto;
  }

  .image-panel img.pixelated {
    image-rendering: pixelated;
  }

  .image-panel span.caption {
    font-size: 12px;
    color: var(--text-dim);
    font-weight: 500;
  }

  /* Detail Sidebar Cards */
  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .info-block {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 16px;
  }

  .info-block h4 {
    margin: 0 0 12px 0;
    font-size: 13px;
    color: var(--cyan);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 13px;
  }

  .metric-row:last-child {
    border-bottom: none;
  }

  .metric-label {
    color: var(--text-dim);
  }

  .metric-val {
    font-weight: 600;
    color: var(--text-bright);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }

  .math-formula {
    background: var(--code-bg);
    padding: 10px 12px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12px;
    color: var(--amber);
    margin: 8px 0;
    border: 1px solid rgba(251, 191, 36, 0.2);
    overflow-x: auto;
  }

  .explanation-text {
    font-size: 13px;
    color: var(--text);
    line-height: 1.6;
    margin: 8px 0 0 0;
  }

  /* Top Peaks Table */
  table.peaks-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    margin-top: 8px;
  }

  table.peaks-table th, table.peaks-table td {
    padding: 6px 8px;
    text-align: left;
    border-bottom: 1px solid var(--card-border);
  }

  table.peaks-table th {
    color: var(--text-dim);
    font-weight: 600;
  }

  table.peaks-table tr.winner {
    background: rgba(52, 211, 153, 0.1);
    color: var(--emerald);
    font-weight: 600;
  }

  .channel-pill-group {
    display: flex;
    gap: 6px;
    margin-top: 10px;
  }

  .chan-pill {
    padding: 3px 8px;
    font-size: 11px;
    border-radius: 4px;
    background: var(--card-border);
    cursor: pointer;
    color: var(--text);
  }

  .chan-pill.active {
    background: var(--accent);
    color: #fff;
  }
</style>
</head>
<body>

<header>
  <div>
    <h1>
      <span>🔬 DriftSenseNet</span>
      <span class="badge">Model Activation Flow</span>
    </h1>
    <span style="font-size: 13px; color: var(--text-dim);">
      Inspecting how raw SEM pictures transform through Siamese Trunk, Grouped Correlation, Wide-RF Context, and Dilated Lattice Head.
    </span>
  </div>
  <div class="header-actions">
    <a href="semicon-driftsense/docs/archify/driftsense-model.architecture.html" class="btn btn-primary" target="_blank">
      🗺️ Open Archify Map
    </a>
    <button class="btn" id="themeToggle" onclick="toggleTheme()">🌓 Toggle Theme</button>
  </div>
</header>

<div class="container">

  <!-- Case Selection Cards -->
  <div class="case-tabs" id="caseTabs">
    <!-- Rendered dynamically -->
  </div>

  <!-- Stage Stepper -->
  <div class="stepper" id="stepper">
    <!-- Rendered dynamically -->
  </div>

  <!-- Workspace Grid -->
  <div class="workspace-grid">
    <!-- Left: Stage Media Canvas -->
    <div class="viewer-card">
      <div class="viewer-header">
        <div class="viewer-title">
          <h2 id="stageTitle">Stage Title</h2>
          <p id="stageSubtitle">Stage explanation</p>
        </div>
        <div id="channelControls" style="display: none;">
          <div class="channel-pill-group" id="channelButtons"></div>
        </div>
      </div>

      <div class="stage-media-container" id="mediaContainer">
        <!-- Rendered dynamically -->
      </div>
    </div>

    <!-- Right: Numerical Breakdown & Theory -->
    <div class="sidebar">
      <div class="info-block">
        <h4>📐 Layer Architecture</h4>
        <div class="metric-row">
          <span class="metric-label">Tensor Shape</span>
          <span class="metric-val" id="metricShape">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Spatial Stride</span>
          <span class="metric-val" id="metricStride">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Effective RF</span>
          <span class="metric-val" id="metricRF">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Channel Count</span>
          <span class="metric-val" id="metricChannels">-</span>
        </div>
      </div>

      <div class="info-block">
        <h4>⚙️ Mathematical Operation</h4>
        <div class="math-formula" id="mathFormula">-</div>
        <p class="explanation-text" id="stageExplanation">-</p>
      </div>

      <div class="info-block" id="predictionBlock">
        <h4>🎯 Localization Result</h4>
        <div class="metric-row">
          <span class="metric-label">Verdict</span>
          <span class="metric-val" id="predFound">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Confidence Score</span>
          <span class="metric-val" id="predConf">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Native Coordinates</span>
          <span class="metric-val" id="predCoord">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Ground Truth</span>
          <span class="metric-val" id="gtCoord">-</span>
        </div>
        <div class="metric-row">
          <span class="metric-label">Localization Error</span>
          <span class="metric-val" id="locError">-</span>
        </div>
      </div>

      <div class="info-block" id="decoyBlock">
        <h4>📊 Candidate Peaks (Decoys)</h4>
        <table class="peaks-table">
          <thead>
            <tr>
              <th>Peak</th>
              <th>Score</th>
              <th>Coord (x, y)</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="peaksTableBody">
          </tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
const TRACES = __JSON_DATA__;

let currentCaseIndex = 0;
let currentStage = 0;
let activeCorrChannel = "all";

const STAGES = [
  {
    id: "inputs",
    name: "Input SEM Images",
    subtitle: "Raw 1 nm/px Reference vs 10 nm/px Search Frame",
    rf: "1 px",
    stride: "1x",
    formula: "Reference (1000x1000 @ 1nm) | Search (1000x1000 @ 10nm)",
    explain: "The Reference is high-dose 1 nm/px with sharp edge definition. The Search frame is captured at 10 nm/px with unknown zoom (z in [8,12]) and rotation (theta in [-5, 5] deg), plus Poisson detector noise.",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.ref_sem}" alt="Reference SEM">
        <span class="caption">Reference Image (1 nm/px, 1000x1000)</span>
      </div>
      <div class="image-panel">
        <img src="${c.images.search_sem}" alt="Search SEM">
        <span class="caption">Search Frame (10 nm/px, native pose: scale ${c.pose.scale.toFixed(2)}x, rot ${c.pose.theta.toFixed(2)}°)</span>
      </div>
    `
  },
  {
    id: "downsampling",
    name: "10x Area Downsampling",
    subtitle: "Creating the 100x100 Template & Canonical Search",
    rf: "10 nm",
    stride: "10x",
    formula: "Template = AreaDownsample(Ref, 10x) -> 100x100 px",
    explain: "Since reference is 1 nm/px and search is 10 nm/px, the reference footprint in search space is exactly 100x100 px. Area interpolation downsamples the reference 10x without aliasing. Both inputs are standardized (mean=0, std=1).",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.template_downsampled}" class="pixelated" style="width: 260px; height: 260px;" alt="Downsampled Template">
        <span class="caption">Nominal 100x100 Template (10 nm/px)</span>
      </div>
      <div class="image-panel">
        <img src="${c.images.canonical_search}" style="width: 380px; height: 380px;" alt="Canonical Search">
        <span class="caption">Canonicalized Search Frame (Posed to scale 10x, 0°)</span>
      </div>
    `
  },
  {
    id: "stem_stride2",
    name: "Encoder Stem /2",
    subtitle: "Conv 7x7 s=2 -> Spatial Stride 2 Feature Map",
    rf: "7 px",
    stride: "2x",
    formula: "Stem1 = ReLU(BN(Conv2d(1, 48, k=7, s=2)))",
    explain: "The shared Siamese trunk processes both template and search frames. First layer uses 7x7 kernel with stride 2 to quickly downsample high-frequency SEM noise while doubling channel capacity.",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.encoder_stem_stride2}" alt="Stem Stride 2">
        <span class="caption">Stem Activation (/2 Stride, 48 channels, mean projection)</span>
      </div>
    `
  },
  {
    id: "stem_stride4",
    name: "Encoder Stem /4",
    subtitle: "Conv 3x3 s=2 -> Spatial Stride 4 Feature Map",
    rf: "15 px",
    stride: "4x",
    formula: "Stem2 = ReLU(BN(Conv2d(48, 96, k=3, s=2)))",
    explain: "Second downsampling stage brings total stride to 4. Template reduces to (96, 25, 25) and Search reduces to (96, Hs/4, Ws/4). Stride 4 balances spatial localization precision against cross-correlation computational cost.",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.encoder_stem_stride4}" alt="Stem Stride 4">
        <span class="caption">Stem Activation (/4 Stride, 96 channels)</span>
      </div>
    `
  },
  {
    id: "dilated_body",
    name: "Dilated ResBlocks & L2 Norm",
    subtitle: "ResBlocks with d=1 and d=2 to preserve local detail",
    rf: "31 px",
    stride: "4x",
    formula: "feat = F.normalize(ResBlock(d=2)(ResBlock(d=1)(stem)), dim=1)",
    explain: "Residual blocks apply dilated convolutions (dilation=2) to expand the receptive field without further downsampling. Channel L2-normalization ensures template and search features represent invariant structural patterns regardless of SEM beam dose or gamma.",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.encoder_dilated_body}" alt="Encoder Body">
        <span class="caption">Normalized Feature Embedding (tf: 25x25, sf: Hs/4 x Ws/4)</span>
      </div>
    `
  },
  {
    id: "grouped_xcorr",
    name: "Grouped Cross-Correlation",
    subtitle: "8 Groups Depthwise Match Volume: Revealing Decoy Lattice",
    rf: "100 px (template size)",
    stride: "4x",
    formula: "Corr = GroupedConv2d(search_feat, template_feat, groups=8)",
    explain: "Template features serve as the convolutional filter across the search embedding. 8 channel groups compute multi-band match scores. Notice the grid of periodic false peaks (the decoy lattice) caused by repeating semiconductor memory cells!",
    render: (c) => {
      const src = activeCorrChannel === 'ch0' ? c.images.raw_corr_ch0 : activeCorrChannel === 'ch3' ? c.images.raw_corr_ch3 : activeCorrChannel === 'ch7' ? c.images.raw_corr_ch7 : c.images.raw_correlation_volume;
      return `
        <div class="image-panel">
          <img id="corrImg" src="${src}" alt="Correlation Volume">
          <span class="caption" id="corrCaption">Grouped Correlation Match Volume (8 groups) — Note the periodic decoy peaks!</span>
        </div>
      `;
    }
  },
  {
    id: "context_branch",
    name: "Wide-RF Context Branch",
    subtitle: "Stacked Dilations (d=2, 4, 8, 16) Reaching >300px RF",
    rf: "> 300 search px",
    stride: "4x",
    formula: "Ctx = Conv(d=16)(Conv(d=8)(Conv(d=4)(Conv(d=2)(sf)))))",
    explain: "The layout repeats locally, but large-scale array mats and flat routing strips are globally unique. The Context Branch uses exponential dilations (2, 4, 8, 16) to see hundreds of pixels across the die, capturing the macro layout cues needed to break the symmetry.",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.context_dilation_2}" alt="Context Dilation 2">
        <span class="caption">Dilation 2 (Local Macro Boundaries)</span>
      </div>
      <div class="image-panel">
        <img src="${c.images.context_dilation_16}" alt="Context Dilation 16">
        <span class="caption">Dilation 16 (Wide-RF Mat & Strip Structure, RF > 300px)</span>
      </div>
    `
  },
  {
    id: "dilated_head",
    name: "Deep Dilated Head",
    subtitle: "Dilations d=1, 2, 4, 8 Over Response Map (Lattice Reasoning)",
    rf: "Whole Decoy Lattice",
    stride: "4x",
    formula: "Head = Conv(d=8)(Conv(d=4)(Conv(d=2)(Conv(d=1)([Corr, Ctx])))))",
    explain: "Crucial design insight: The head is dilated OVER THE RESPONSE MAP (d=1,2,4,8). Instead of evaluating each candidate peak in isolation, it evaluates the peak against the entire periodic decoy lattice and macro context, suppressing secondary peaks and elevating the true match!",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.dilated_head_lattice}" alt="Dilated Head">
        <span class="caption">Dilated Head Response Feature Map (96 channels)</span>
      </div>
    `
  },
  {
    id: "output_heads",
    name: "Center Heatmap & Offset Field",
    subtitle: "Sub-Cell Continuous Nanometer Coordinate Localization",
    rf: "Final Decision",
    stride: "Continuous",
    formula: "Coord = ((j + dx)*4 + 50, (i + dy)*4 + 50) -> Native Uncanonicalize",
    explain: "Logit head predicts 1-channel probability heatmap with low prior bias (-4.0). Offset head predicts continuous fractional shifts (dx, dy) in stride units. Continuous coordinate is uncanonicalized back to native search frame with sub-pixel precision!",
    render: (c) => `
      <div class="image-panel">
        <img src="${c.images.heatmap_probability}" alt="Heatmap Probability">
        <span class="caption">Predicted Match Probability Heatmap (Sigmoid Peak: ${c.prediction.confidence.toFixed(4)})</span>
      </div>
    `
  }
];

function init() {
  renderCaseTabs();
  renderStepper();
  updateView();
}

function renderCaseTabs() {
  const container = document.getElementById("caseTabs");
  container.innerHTML = TRACES.map((c, idx) => `
    <div class="case-tab ${idx === currentCaseIndex ? 'active' : ''}" onclick="selectCase(${idx})">
      <h3>
        <span>${c.title}</span>
        <span class="status-pill ${c.prediction.found ? 'pass' : 'reject'}">
          ${c.prediction.found ? 'FOUND (' + (c.prediction.confidence * 100).toFixed(1) + '%)' : 'REJECTED (' + (c.prediction.confidence * 100).toFixed(2) + '%)'}
        </span>
      </h3>
      <p>${c.description}</p>
    </div>
  `).join("");
}

function renderStepper() {
  const container = document.getElementById("stepper");
  container.innerHTML = STAGES.map((s, idx) => `
    <button class="step-btn ${idx === currentStage ? 'active' : ''}" onclick="selectStage(${idx})">
      <span class="num">STAGE ${idx + 1}</span>
      ${s.name}
    </button>
  `).join("");
}

function selectCase(idx) {
  currentCaseIndex = idx;
  renderCaseTabs();
  updateView();
}

function selectStage(idx) {
  currentStage = idx;
  renderStepper();
  updateView();
}

function setCorrChannel(ch) {
  activeCorrChannel = ch;
  updateView();
}

function updateView() {
  const c = TRACES[currentCaseIndex];
  const s = STAGES[currentStage];

  document.getElementById("stageTitle").innerText = `Stage ${currentStage + 1}: ${s.name}`;
  document.getElementById("stageSubtitle").innerText = s.subtitle;
  document.getElementById("metricShape").innerText = getStageShape(c, s.id);
  document.getElementById("metricStride").innerText = s.stride;
  document.getElementById("metricRF").innerText = s.rf;
  document.getElementById("metricChannels").innerText = getStageChannels(c, s.id);
  document.getElementById("mathFormula").innerText = s.formula;
  document.getElementById("stageExplanation").innerText = s.explain;

  // Media container
  const mediaContainer = document.getElementById("mediaContainer");
  mediaContainer.innerHTML = s.render(c);

  // Channel controls for cross-correlation
  const chanCtrl = document.getElementById("channelControls");
  if (s.id === "grouped_xcorr") {
    chanCtrl.style.display = "block";
    const btns = document.getElementById("channelButtons");
    btns.innerHTML = `
      <span class="chan-pill ${activeCorrChannel === 'all' ? 'active' : ''}" onclick="setCorrChannel('all')">Mean (All 8)</span>
      <span class="chan-pill ${activeCorrChannel === 'ch0' ? 'active' : ''}" onclick="setCorrChannel('ch0')">Group 0</span>
      <span class="chan-pill ${activeCorrChannel === 'ch3' ? 'active' : ''}" onclick="setCorrChannel('ch3')">Group 3</span>
      <span class="chan-pill ${activeCorrChannel === 'ch7' ? 'active' : ''}" onclick="setCorrChannel('ch7')">Group 7</span>
    `;
  } else {
    chanCtrl.style.display = "none";
  }

  // Prediction block
  const p = c.prediction;
  document.getElementById("predFound").innerHTML = p.found
    ? '<span style="color: var(--emerald);">FOUND (Match Confirmed)</span>'
    : '<span style="color: var(--rose);">ABSENT (Correctly Declined)</span>';
  document.getElementById("predConf").innerText = `${p.confidence.toFixed(4)} (threshold: 0.1800)`;
  document.getElementById("predCoord").innerText = `(${p.pred_x.toFixed(2)}, ${p.pred_y.toFixed(2)})`;
  
  if (c.ground_truth.found) {
    document.getElementById("gtCoord").innerText = `(${c.ground_truth.gt_x.toFixed(2)}, ${c.ground_truth.gt_y.toFixed(2)})`;
    document.getElementById("locError").innerHTML = `<strong style="color: var(--emerald);">${p.error_px.toFixed(3)} px</strong> (Sub-pixel recovery!)`;
  } else {
    document.getElementById("gtCoord").innerText = "None (Independent Die Region)";
    document.getElementById("locError").innerText = "N/A (True Negative)";
  }

  // Candidate peaks table
  const tbody = document.getElementById("peaksTableBody");
  tbody.innerHTML = c.top_peaks.map((pk, idx) => `
    <tr class="${idx === 0 ? 'winner' : ''}">
      <td>${idx === 0 ? '🏆 Winner' : 'Decoy ' + idx}</td>
      <td>${pk.score.toFixed(4)}</td>
      <td>(${pk.native_xy[0].toFixed(1)}, ${pk.native_xy[1].toFixed(1)})</td>
      <td>${idx === 0 ? (p.found ? 'Accepted' : 'Suppressed') : 'Suppressed'}</td>
    </tr>
  `).join("");
}

function getStageShape(c, stageId) {
  switch(stageId) {
    case "inputs": return `Ref: [${c.shapes.ref_shape.join(", ")}] | Search: [${c.shapes.search_shape.join(", ")}]`;
    case "downsampling": return `Template: [${c.shapes.template_shape.join(", ")}] | Canon: [${c.shapes.canonical_search_shape.join(", ")}]`;
    case "stem_stride2": return `[48, H/2, W/2]`;
    case "stem_stride4": return `[96, H/4, W/4]`;
    case "dilated_body": return `Template: [${c.shapes.encoder_template_feat.join(", ")}] | Search: [${c.shapes.encoder_search_feat.join(", ")}]`;
    case "grouped_xcorr": return `Volume: [${c.shapes.correlation_volume.join(", ")}]`;
    case "context_branch": return `Context: [${c.shapes.context_features.join(", ")}]`;
    case "dilated_head": return `Head: [${c.shapes.dilated_head_features.join(", ")}]`;
    case "output_heads": return `Logit: [1, Hc, Wc] | Offset: [2, Hc, Wc]`;
    default: return "-";
  }
}

function getStageChannels(c, stageId) {
  switch(stageId) {
    case "inputs": return "1 (Grayscale)";
    case "downsampling": return "1 (Grayscale standardized)";
    case "stem_stride2": return "48 channels";
    case "stem_stride4": return "96 channels";
    case "dilated_body": return "96 channels (L2-norm)";
    case "grouped_xcorr": return "8 match groups (raw) -> 96 mixed";
    case "context_branch": return "48 context channels";
    case "dilated_head": return "144 fused -> 96 channels";
    case "output_heads": return "1 logit + 2 offsets (dx, dy)";
    default: return "-";
  }
}

function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || "dark";
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
}

window.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


def build_html():
    print(f"Reading traces from {TRACES_PATH}...")
    with open(TRACES_PATH, "r") as f:
        traces = json.load(f)

    json_str = json.dumps(traces)
    html_content = TEMPLATE_HTML.replace("__JSON_DATA__", json_str)

    with open(OUTPUT_HTML_PROJECT, "w") as f:
        f.write(html_content)
    print(f"Generated project HTML: {OUTPUT_HTML_PROJECT} ({os.path.getsize(OUTPUT_HTML_PROJECT):,} bytes)")

    with open(OUTPUT_HTML_ROOT, "w") as f:
        f.write(html_content)
    print(f"Generated root HTML: {OUTPUT_HTML_ROOT} ({os.path.getsize(OUTPUT_HTML_ROOT):,} bytes)")


if __name__ == "__main__":
    build_html()
