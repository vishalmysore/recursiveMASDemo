import * as webllm from '@mlc-ai/web-llm';
import { loadRecursiveLinks } from './recursive-link.js';
import { getLatentRuntime, latentForward, cosine } from './latent-core.js';
import { chainForward, chainDecode } from './latent-chain.js';

/* ════════════════════════════════════════════════════════════════════════════
   RecursiveMAS Playground
   An interactive, in-browser demo of "Recursive Multi-Agent Systems".

   Real local LLMs (WebLLM/WebGPU) are chained into a recursive collaboration
   loop. Two execution modes are compared:

     • RecursiveMAS (latent) — intermediate agents emit a compact "latent
       thought" instead of full prose; only the final agent in the final round
       decodes a full text answer. (A faithful, browser-runnable approximation
       of the paper's latent-state passing — WebLLM can't expose true hidden
       states, so we compress the carried text instead.)

     • Text-MAS (baseline)   — every agent, every round, decodes full reasoning
       text that is passed wholesale to the next agent. (The paper's
       Recursive-TextMAS baseline.)

   The token/time gap between the two reproduces the paper's headline result.
   ════════════════════════════════════════════════════════════════════════════ */

// ── Models ───────────────────────────────────────────────────────────────────
// Only the custom latent-exposing backbone is offered. The RecursiveMAS-0.5B record
// is appended below (from CUSTOM_MODELS) and becomes the sole, default-selected option.
const MODELS = [];

// ── Custom (self-compiled) models ────────────────────────────────────────────
// Drop a record here once the model builder repo has published a model — i.e. one
// whose graph exposes internal latent (last-layer hidden) state for true latent
// transfer. The model is compiled in a SEPARATE repo:
//   https://github.com/vishalmysore/recursiveMASWebLLM
// which publishes the .wasm to a GitHub Release and the weights to Hugging Face.
// Each entry here is auto-registered with WebLLM (merged into appConfig) and added
// to the backbone picker.
//
// Fields: `model` = HF weights repo · `model_id` = your id · `model_lib` = Release
// .wasm URL · `label`/`size` = picker display (UI only) · `exposesLatent` = mark it
// latent-transfer capable (UI only) · `recursiveLink` = trained W1/W2/W3 JSON URL.
const CUSTOM_MODELS = [
  {
    // Self-compiled model, published to Hugging Face (note the case: VishalMysore).
    model:    'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC',
    model_id: 'recursivemas-0.5b',
    // ?v=N is a cache-buster: WebLLM caches the model lib in Cache Storage keyed by
    // this exact URL and never re-validates it. BUMP N whenever the .wasm is rebuilt,
    // or stale libs (missing get_last_hidden) get served from cache forever.
    model_lib:'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC/resolve/main/libs/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm?v=2',
    vram_required_MB: 900,
    label: 'RecursiveMAS 0.5B', size: '~0.5 GB · custom', exposesLatent: true,
  },
];

// Merge our custom records into WebLLM's prebuilt app config (so both prebuilt and
// custom models resolve). Strips UI-only fields before handing records to WebLLM.
function buildAppConfig() {
  if (!CUSTOM_MODELS.length) return webllm.prebuiltAppConfig;
  const records = CUSTOM_MODELS.map(({ label, size, exposesLatent, ...rec }) => rec);
  return {
    ...webllm.prebuiltAppConfig,
    model_list: [...webllm.prebuiltAppConfig.model_list, ...records],
  };
}

// Make custom models selectable + label-resolvable alongside the built-ins.
CUSTOM_MODELS.forEach(c => MODELS.push({
  id: c.model_id, label: c.label || c.model_id, size: c.size, custom: true,
  exposesLatent: c.exposesLatent, recursiveLink: c.recursiveLink,
}));

// Holds the trained RecursiveLink (W1/W2/W3) once a latent-capable model is picked.
// The link math runs in-browser (recursive-link.js); the low-level hidden-state loop
// that would consume it is documented there and in model-build/ (the remaining piece).
let recursiveLinks = null;
async function maybeLoadLink() {
  const m = MODELS.find(x => x.id === getModelId());
  if (m?.exposesLatent && m.recursiveLink && !recursiveLinks) {
    try {
      recursiveLinks = await loadRecursiveLinks(m.recursiveLink);
      console.info(`[RecursiveLink] loaded ${recursiveLinks.links.length} link(s), hidden=${recursiveLinks.hidden}`);
    } catch (e) { console.warn('[RecursiveLink] load failed:', e.message); }
  }
}

// ── Collaboration patterns (from the paper, Section 2 + Table 1) ──────────────
// `paperModel` is the heterogeneous model the paper assigns to each role; shown
// as a node sub-label even though one backbone runs every role in this demo.
const PATTERNS = {
  sequential: {
    label: 'Sequential', emoji: '🔗',
    desc: 'Chain-of-agents. A Planner decomposes the problem, a Critic judges and pokes holes, and a Solver produces the answer — refined over each recursion round.',
    flow: 'linear',
    agents: [
      { key: 'planner', role: 'Planner', emoji: '🗺️', paperModel: 'Qwen3-1.7B',
        prompt: 'You are the PLANNER in a multi-agent team. Decompose the problem into the key sub-steps and the approach to take. Do NOT solve it fully — lay out the plan and the critical quantities/cases to handle.' },
      { key: 'critic', role: 'Critic', emoji: '🔍', paperModel: 'Llama3.2-1B',
        prompt: 'You are the CRITIC. Examine the upstream plan/work for errors, missing cases, and wrong assumptions. State concretely what must be fixed or verified. Be sharp and specific.' },
      { key: 'solver', role: 'Solver', emoji: '🧮', paperModel: 'Qwen2.5-Math-1.5B',
        prompt: 'You are the SOLVER. Using the plan and the critic\'s notes, carry out the work and produce the answer. Show the decisive steps and end with a clear final result.' },
    ],
  },
  mixture: {
    label: 'Mixture', emoji: '🧩',
    desc: 'A mixture of domain specialists (Math, Code, Science) each attack the problem in parallel; a Summarizer aggregates their views into one answer.',
    flow: 'mixture',
    agents: [
      { key: 'math', role: 'Math Specialist', emoji: '➗', paperModel: 'DeepSeek-R1-Distill-Qwen-1.5B', parallel: true,
        prompt: 'You are the MATH SPECIALIST. Analyze the problem from a quantitative/mathematical angle and contribute the math-relevant insight or computation. Stay in your lane.' },
      { key: 'code', role: 'Code Specialist', emoji: '💻', paperModel: 'Qwen2.5-Coder-3B', parallel: true,
        prompt: 'You are the CODE SPECIALIST. Contribute an algorithmic/implementation view: how you would compute or verify the answer in code, edge cases, and complexity. Stay in your lane.' },
      { key: 'science', role: 'Science Specialist', emoji: '🔬', paperModel: 'BioMistral-7B', parallel: true,
        prompt: 'You are the SCIENCE SPECIALIST. Contribute the scientific/conceptual reasoning and any domain facts or principles that bear on the problem. Stay in your lane.' },
      { key: 'summarizer', role: 'Summarizer', emoji: '🧷', paperModel: 'Qwen3.5-2B',
        prompt: 'You are the SUMMARIZER. You receive the specialists\' contributions. Reconcile them, resolve disagreements, and produce one clear final answer.' },
    ],
  },
  distillation: {
    label: 'Distillation', emoji: '🎓',
    desc: 'A larger, more capable Expert distills its reasoning to a smaller, faster Learner, which produces the final answer efficiently.',
    flow: 'linear',
    agents: [
      { key: 'expert', role: 'Expert', emoji: '🧠', paperModel: 'Qwen3.5-9B',
        prompt: 'You are the EXPERT (a larger model). Provide high-quality guidance: the key reasoning, the method, and the pitfalls — distilled so a smaller learner can follow it. Do not pad.' },
      { key: 'learner', role: 'Learner', emoji: '🐣', paperModel: 'Qwen3.5-4B',
        prompt: 'You are the LEARNER (a smaller, faster model). Follow the expert\'s distilled guidance and produce the final answer concisely.' },
    ],
  },
  deliberation: {
    label: 'Deliberation', emoji: '🛠️',
    desc: 'An inner-thinking Reflector debates with a Tool-Caller that can invoke external tools (here, a real Wikipedia search). They iterate to consensus; the Tool-Caller produces the final answer.',
    flow: 'linear',
    tools: true,
    agents: [
      { key: 'reflector', role: 'Reflector', emoji: '🪞', paperModel: 'Qwen3.5-4B',
        prompt: 'You are the REFLECTOR. Think through the problem, identify exactly what factual information is still missing, and state the single most useful query the tool-caller should look up.' },
      { key: 'toolcaller', role: 'Tool-Caller', emoji: '🛰️', paperModel: 'Qwen3.5-4B (+tools)',
        prompt: 'You are the TOOL-CALLER. You can look facts up. If you need a fact, write exactly one line: SEARCH_WEB(your query). Otherwise, use what is known to produce the final answer.' },
    ],
  },
};

// ── Task presets ──────────────────────────────────────────────────────────────
const TASK_PRESETS = {
  sequential: [
    'What is the smallest positive integer n such that n! is divisible by 990?',
    'A train leaves at 60 mph; another leaves the same station 30 min later at 75 mph. How far from the station do they meet?',
  ],
  mixture: [
    'Estimate how many piano tuners work in Chicago, and explain the reasoning.',
    'Is 1,000,003 prime? Justify the answer and describe how you would verify it.',
  ],
  distillation: [
    'Explain why the sky is blue, then give a one-sentence summary a 10-year-old would understand.',
    'Write a Python function that returns the longest palindromic substring of a string.',
  ],
  deliberation: [
    'Which country has won the most FIFA World Cup titles, and how many?',
    'Who developed the theory of general relativity and in what year was it published?',
  ],
};

// ── Tunables ──────────────────────────────────────────────────────────────────
const MAXTOK_LATENT = 70;    // intermediate "latent thought" budget
const MAXTOK_TEXT   = 256;   // full reasoning text (baseline + text mode)
const MAXTOK_DECODE = 340;   // final decoded answer
const TEMP          = 0.6;

// Embedding memory bus: a real on-device embedding model (snowflake-arctic-embed-s,
// ~33M) lets agents route state to each other through a shared vector store instead
// of dumping prose down the chain. The embeddings genuinely flow between agents as
// the addressing layer; what finally enters the LLM is the retrieved text (the only
// thing a chat model can ingest).
const EMBED_MODEL = 'snowflake-arctic-embed-s-q0f32-MLC-b4'; // -b4 = batch-4 build (embeds one text at a time)
const BUS_K       = 3;   // how many prior latents each agent retrieves

// ── Runtime state ─────────────────────────────────────────────────────────────
let engine        = null;
let loadedSig     = null;    // signature of the loaded model set (chat[/+embed])
let chatModelId   = null;    // currently loaded chat model id
let running       = false;
let cancelled     = false;
let busEnabled    = false;   // embedding memory bus toggle
let latentMemory  = [];      // [{id, round, role, emoji, text, vector}]
const results     = { latent: null, text: null }; // {finalText, totals, transcript}

// Real latent-space runtime: invokes the custom model's get_last_hidden on-device.
let latentRT      = null;    // { ok, ... } resolved after the model loads (latent-core.js)
let prevLatentVec = null;    // previous agent's pooled latent — for latent-space routing cosine
let chainEnabled  = false;   // true vector-only transfer (intermediates never decode text)

// ── DOM helpers ───────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── DOM refs ──────────────────────────────────────────────────────────────────
const patternCards   = $('pattern-cards');
const patternDesc    = $('pattern-desc');
const roundsSlider   = $('rounds-slider');
const roundsVal      = $('rounds-val');
const taskChips      = $('task-chips');
const taskInput      = $('task-input');
const runLatentBtn   = $('run-latent-btn');
const runTextBtn     = $('run-text-btn');
const runBothBtn     = $('run-both-btn');
const stopBtn        = $('stop-btn');
const modelStatusTxt = $('model-status-text');
const modelBadge     = $('model-badge');
const progressWrap   = $('progress-wrap');
const progressBar    = $('progress-bar');
const modelHint      = $('model-hint');
const loopViz        = $('loop-viz');
const vizRound       = $('viz-round');
const metricsCard    = $('metrics-card');
const metricsGrid    = $('metrics-grid');
const metricsDelta   = $('metrics-delta');
const transcriptCard = $('transcript-card');
const tLatent        = $('transcript-latent');
const tText          = $('transcript-text');
const busToggle      = $('bus-toggle');
const busCard        = $('bus-card');
const busList        = $('bus-list');
const busDim         = $('bus-dim');

// ── Selection helpers ─────────────────────────────────────────────────────────
const getPattern = () => document.querySelector('input[name="pattern"]:checked').value;
const getModelId = () => document.querySelector('input[name="model"]:checked').value;
const getRounds  = () => parseInt(roundsSlider.value, 10);

// ════════════════════════════════════════════════════════════════════════════
// UI wiring
// ════════════════════════════════════════════════════════════════════════════
function renderPattern() {
  const p = PATTERNS[getPattern()];
  patternDesc.textContent = p.desc;
  renderTaskChips();
  buildLoopViz();
}

function renderTaskChips() {
  const presets = TASK_PRESETS[getPattern()] ?? [];
  taskChips.innerHTML = '';
  presets.forEach(t => {
    const chip = el('span', 'task-chip', esc(t.length > 52 ? t.slice(0, 50) + '…' : t));
    chip.title = t;
    chip.onclick = () => { taskInput.value = t; };
    taskChips.appendChild(chip);
  });
  if (presets[0] && !taskInput.value) taskInput.value = presets[0];
}

patternCards.addEventListener('change', renderPattern);
roundsSlider.addEventListener('input', () => { roundsVal.textContent = roundsSlider.value; });

// Build the agent-loop diagram (static, before a run).
function buildLoopViz() {
  const p = PATTERNS[getPattern()];
  loopViz.innerHTML = '';
  vizRound.textContent = 'round —';
  const agents = p.agents;
  agents.forEach((a, i) => {
    loopViz.appendChild(makeNode(a));
    if (i < agents.length - 1) loopViz.appendChild(el('div', 'agent-edge', '→'));
  });
  const fb = el('div', 'feedback-arrow', `↺ last agent's latent state loops back to ${agents[0].role} each round (outer + inner RecursiveLink)`);
  loopViz.appendChild(fb);
}

function makeNode(a) {
  const node = el('div', 'agent-node');
  node.dataset.key = a.key;
  node.innerHTML =
    `<div class="node-inner-badge" title="inner RecursiveLink">R<sub>in</sub></div>` +
    `<div class="node-emoji">${a.emoji}</div>` +
    `<div class="node-role">${esc(a.role)}</div>` +
    `<div class="node-model" title="model the paper assigns to this role">paper: ${esc(a.paperModel)}</div>` +
    `<div class="node-state"></div>`;
  return node;
}

function vizReset() {
  loopViz.querySelectorAll('.agent-node').forEach(n => { n.classList.remove('active', 'done', 'decoding'); n.querySelector('.node-state').textContent = ''; });
  loopViz.querySelectorAll('.agent-edge').forEach(e => e.classList.remove('active'));
  loopViz.querySelector('.feedback-arrow')?.classList.remove('active');
}

function vizSetActive(key, { decode = false, state = '' } = {}) {
  const nodes = [...loopViz.querySelectorAll('.agent-node')];
  nodes.forEach(n => {
    if (n.dataset.key === key) {
      n.classList.add('active');
      n.classList.toggle('decoding', decode);
      n.querySelector('.node-state').textContent = state;
    } else if (n.classList.contains('active')) {
      n.classList.remove('active', 'decoding');
      n.classList.add('done');
    }
  });
  // light the edge leading into the active node
  loopViz.querySelectorAll('.agent-edge').forEach(e => e.classList.remove('active'));
  const idx = nodes.findIndex(n => n.dataset.key === key);
  if (idx > 0) loopViz.querySelectorAll('.agent-edge')[idx - 1]?.classList.add('active');
}

// ════════════════════════════════════════════════════════════════════════════
// Model loading
// ════════════════════════════════════════════════════════════════════════════
async function checkWebGPU() {
  if (!navigator.gpu) return false;
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  return !!adapter;
}

function setModelStatus(text, cls) {
  modelStatusTxt.textContent = text;
  modelBadge.className = 'status-badge status-' + cls;
  modelBadge.textContent = cls;
}

async function ensureModel() {
  const wanted = getModelId();
  // When the bus is on we also load the embedding model into the same engine.
  const toLoad = busEnabled ? [wanted, EMBED_MODEL] : [wanted];
  const sig = toLoad.join('|');
  if (engine && loadedSig === sig) return true;

  if (!(await checkWebGPU())) {
    setModelStatus('WebGPU not available', 'error');
    modelHint.innerHTML =
      '<strong style="color:var(--error)">WebGPU is required for WebLLM.</strong><br>' +
      '① Use Chrome/Edge 113+ in a regular (non-incognito) window<br>' +
      '② If needed, enable <code>chrome://flags/#enable-unsafe-webgpu</code><br>' +
      '③ Verify at <a href="https://webgpureport.org" target="_blank" rel="noopener">webgpureport.org</a>';
    return false;
  }

  const label = MODELS.find(m => m.id === wanted)?.label ?? wanted;
  progressWrap.style.display = 'block';
  setModelStatus(`Loading ${label}${busEnabled ? ' + embedder' : ''}…`, 'loading');

  if (engine) { try { await engine.unload(); } catch {} engine = null; loadedSig = null; }

  engine = await webllm.CreateMLCEngine(toLoad, {
    appConfig: buildAppConfig(),
    initProgressCallback: (p) => {
      const pct = Math.round((p.progress ?? 0) * 100);
      progressBar.style.width = pct + '%';
      setModelStatus(p.text ?? `Loading… ${pct}%`, 'loading');
    },
  });
  chatModelId = wanted;
  loadedSig = sig;

  // Resolve the real latent runtime for latent-capable backbones (get_last_hidden).
  const m = MODELS.find(x => x.id === wanted);
  latentRT = m?.exposesLatent ? getLatentRuntime(engine, wanted) : { ok: false, reason: 'model not marked latent-capable' };
  console.info(latentRT.ok
    ? '[latent] get_last_hidden runtime ready — agents will compute real last-layer hidden states'
    : `[latent] real latent unavailable (${latentRT.reason}) — falling back to compressed-text thoughts`);

  progressWrap.style.display = 'none';
  setModelStatus(`${label} ready${busEnabled ? ' · embedder ready' : ''}${latentRT.ok ? ' · latent ✓' : ''}`, 'ready');
  return true;
}

// ════════════════════════════════════════════════════════════════════════════
// Generation
// ════════════════════════════════════════════════════════════════════════════
const estTokens = (s) => Math.ceil((s || '').trim().split(/\s+/).filter(Boolean).length * 1.3);

async function generate({ system, user, maxTokens, stream = false, onToken }) {
  const t0 = performance.now();
  let text = '', usage = null;
  const messages = [{ role: 'system', content: system }, { role: 'user', content: user }];

  if (stream) {
    const chunks = await engine.chat.completions.create({
      model: chatModelId, messages, temperature: TEMP, max_tokens: maxTokens, stream: true,
      stream_options: { include_usage: true },
    });
    for await (const c of chunks) {
      if (cancelled) break;
      const d = c.choices?.[0]?.delta?.content || '';
      if (d) { text += d; onToken?.(d); }
      if (c.usage) usage = c.usage;
    }
  } else {
    const res = await engine.chat.completions.create({ model: chatModelId, messages, temperature: TEMP, max_tokens: maxTokens });
    text = res.choices?.[0]?.message?.content || '';
    usage = res.usage;
  }

  const time = performance.now() - t0;
  const tokens = usage?.total_tokens ?? (estTokens(system) + estTokens(user) + estTokens(text));
  return { text: (text || '').trim(), tokens, time };
}

// ── Embedding memory bus helpers ────────────────────────────────────────────────
async function embed(text) {
  const res = await engine.embeddings.create({ model: EMBED_MODEL, input: (text || '').slice(0, 2000) || ' ' });
  return res.data?.[0]?.embedding ?? [];
}
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
function retrieveTopK(queryVec, k) {
  return latentMemory
    .map(e => ({ ...e, score: cosineSim(queryVec, e.vector) }))
    .sort((x, y) => y.score - x.score)
    .slice(0, k);
}

// Per-role instruction tail that differs between latent and text modes.
function modeTail(mode, decode) {
  if (mode === 'text' || decode) {
    return decode
      ? '\n\nProduce the FINAL answer now — clear, complete, and correct.'
      : '\n\nExplain your full reasoning step by step so the next agent can build on it.';
  }
  // latent intermediate
  return '\n\nOutput ONLY a compact latent thought for the next agent: the essential variables, ' +
         'partial results, and the single most useful insight. No prose, no restating the question, ' +
         'max ~25 words. This is internal state, not an answer.';
}

function buildUser(task, incoming, round, rounds, fromRole) {
  let u = `TASK: ${task}\n`;
  if (round > 1 || incoming) {
    u += `\nRECURSION ROUND ${round} of ${rounds}.`;
  }
  if (incoming) {
    u += `\nUpstream state${fromRole ? ` from the ${fromRole}` : ''}:\n${incoming}`;
  }
  return u;
}

// ════════════════════════════════════════════════════════════════════════════
// Wikipedia search tool (Deliberation pattern) — CORS-enabled, no key.
// ════════════════════════════════════════════════════════════════════════════
async function searchWeb(query) {
  try {
    const url = `https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(query)}&format=json&origin=*&srlimit=3`;
    const r = await fetch(url);
    const j = await r.json();
    const hits = (j.query?.search || []).map(s => `${s.title}: ${s.snippet.replace(/<[^>]+>/g, '')}`);
    return hits.length ? hits.join('  |  ') : 'No results found.';
  } catch (e) {
    return 'Search failed (' + e.message + ').';
  }
}

// ════════════════════════════════════════════════════════════════════════════
// The recursion engine
// ════════════════════════════════════════════════════════════════════════════
async function runMode(mode, task, pattern, rounds, tx) {
  const P = PATTERNS[pattern];
  const totals = { tokens: 0, time: 0, calls: 0 };
  let carry = null;        // latent state looped back from previous round
  let finalText = '';
  if (mode === 'latent') prevLatentVec = null;  // reset latent-space routing chain

  // True vector-only transfer: intermediates never decode text. Linear patterns only
  // (tool hops + the mixture aggregator need text), and requires the latent runtime.
  const canChain = mode === 'latent' && chainEnabled && latentRT?.ok && P.flow !== 'mixture' && !P.tools;
  if (mode === 'latent' && chainEnabled && !canChain) {
    addChainNote(tx, latentRT?.ok
      ? `latent-only transfer skips “${P.label}” (needs text for tools/aggregation) — running classic latent`
      : `latent-only unavailable (${latentRT?.reason || 'no runtime'}) — running classic latent`);
  }
  if (canChain) return runLatentChain(task, P, rounds, tx);

  // The embedding bus only applies to the RecursiveMAS (latent) run, not the text baseline.
  const useBus = busEnabled && mode === 'latent';

  for (let r = 1; r <= rounds && !cancelled; r++) {
    vizRound.textContent = `round ${r} / ${rounds}`;
    addRoundDivider(tx, r, rounds);
    const isFinalRound = r === rounds;

    if (useBus) {
      ({ carry, finalText } = await runBusRound(task, P, r, rounds, isFinalRound, totals, tx));
    } else if (P.flow === 'mixture') {
      ({ carry, finalText } = await runMixtureRound(mode, task, P, r, rounds, carry, isFinalRound, totals, tx));
    } else {
      ({ carry, finalText } = await runLinearRound(mode, task, P, r, rounds, carry, isFinalRound, totals, tx));
    }
    if (r < rounds) { loopViz.querySelector('.feedback-arrow')?.classList.add('active'); await sleep(450); loopViz.querySelector('.feedback-arrow')?.classList.remove('active'); }
  }
  return { finalText, totals, transcript: tx };
}

// ── True vector-only RecursiveMAS (latent-chain.js) ──────────────────────────────
// Intermediate agents forward through get_last_hidden over the previous agent's
// injected latent and emit a hidden vector (never text). The accumulated latent loops
// back to the first agent each round; only the final agent in the final round decodes.
// Falls back to the classic latent run if any low-level step fails.
async function runLatentChain(task, P, rounds, tx) {
  const totals = { tokens: 0, time: 0, calls: 0 };
  const agents = P.agents;
  let latent = null;      // pooled latent vector carried between agents / rounds
  let finalText = '';
  prevLatentVec = null;

  for (let r = 1; r <= rounds && !cancelled; r++) {
    vizRound.textContent = `round ${r} / ${rounds}`;
    addRoundDivider(tx, r, rounds);
    const isFinalRound = r === rounds;

    for (let i = 0; i < agents.length && !cancelled; i++) {
      const a = agents[i];
      const isLastAgent = i === agents.length - 1;
      const decode = isFinalRound && isLastAgent;
      const prompt = `${a.prompt}\n\n${buildUser(task, null, r, rounds, latent ? 'the previous agent (latent)' : null)}`;

      vizSetActive(a.key, { decode, state: decode ? 'decoding ← latent' : 'latent (get_last_hidden)' });
      await sleep(120);

      const t0 = performance.now();
      if (decode) {
        const msg = addMsg(tx, a.emoji, a.role, 'decode', '', 'decode');
        const body = msg.querySelector('.msg-body');
        const res = await chainDecode(latentRT, prompt, latent, {
          maxTokens: MAXTOK_DECODE, temperature: TEMP,
          onToken: (d) => { body.textContent += d; body.scrollIntoView({ block: 'nearest' }); },
          isStopped: () => cancelled,
        });
        if (!res.ok) {
          addChainNote(tx, `decode failed at ${res.stage}: ${res.error} — falling back to classic latent decode`);
          const out = await generate({ system: a.prompt, user: buildUser(task, null, r, rounds, null) + modeTail('latent', true), maxTokens: MAXTOK_DECODE, stream: true, onToken: (d) => { body.textContent += d; } });
          finalText = out.text; totals.tokens += out.tokens; totals.time += out.time;
        } else {
          if (!body.textContent) body.textContent = res.text || '(no output)';
          finalText = res.text;
          totals.tokens += res.nTokens;
          addChainProof(msg, { decoded: true, injected: res.injected, nTokens: res.nTokens });
        }
        totals.time += performance.now() - t0;
        totals.calls += 1;
      } else {
        const msg = addMsg(tx, a.emoji, a.role, 'latent', '⟶ latent only (no text decode)', 'latent');
        const res = await chainForward(latentRT, prompt, latent);
        totals.time += performance.now() - t0;
        totals.calls += 1;
        if (!res.ok) {
          addChainNote(tx, `${a.role} latent forward failed at ${res.stage}: ${res.error} — aborting latent-only run`);
          return { finalText, totals, transcript: tx, aborted: true };
        }
        const c = res.pooled && prevLatentVec ? cosine(prevLatentVec, res.pooled) : null;
        if (res.pooled) { latent = res.pooled; prevLatentVec = res.pooled; }
        addChainProof(msg, { shape: res.shape, dtype: res.dtype, norm: res.norm, injected: res.injected, cos: c });
        msg.querySelector('.msg-meta').textContent = `${(((performance.now() - t0)) / 1000).toFixed(1)}s · no decode`;
      }
    }
    if (r < rounds) { loopViz.querySelector('.feedback-arrow')?.classList.add('active'); await sleep(450); loopViz.querySelector('.feedback-arrow')?.classList.remove('active'); }
  }
  return { finalText, totals, transcript: tx };
}

// Embedding-bus round: every agent retrieves the most relevant prior latents from a
// shared vector store (real on-device embeddings) instead of receiving the linear
// hand-off, then writes its own latent back into the store.
async function runBusRound(task, P, r, rounds, isFinalRound, totals, tx) {
  const agents = P.agents;
  let lastOut = '';

  for (let i = 0; i < agents.length && !cancelled; i++) {
    const a = agents[i];
    const isLastAgent = i === agents.length - 1;
    const decode = isFinalRound && isLastAgent;

    // 1) retrieve relevant prior latents by embedding similarity
    const qVec = await embed(`${a.role} working on: ${task}`);
    if (cancelled) break;
    totals.embedCalls = (totals.embedCalls || 0) + 1;
    const hits = retrieveTopK(qVec, BUS_K);
    addRetrievalNote(tx, a, hits);

    vizSetActive(a.key, { decode, state: decode ? 'decode → text' : 'latent (bus)' });
    await sleep(120);

    const incoming = hits.length
      ? hits.map(h => `• [${h.role}, round ${h.round}, sim ${h.score.toFixed(2)}] ${h.text}`).join('\n')
      : null;
    const fromRole = hits.length ? 'the latent memory bus' : null;
    const user = buildUser(task, incoming, r, rounds, fromRole) + modeTail('latent', decode);
    const out = await runAgentCall({ a, system: a.prompt, user, decode, mode: 'latent', totals, tx });
    if (cancelled) break;

    // 2) write this agent's latent into the shared store
    const vec = await embed(out.text);
    totals.embedCalls += 1;
    const entry = { id: `${r}-${a.key}-${Date.now()}`, round: r, role: a.role, emoji: a.emoji, text: out.text, vector: vec };
    latentMemory.push(entry);
    addBusEntry(entry, vec.length);
    lastOut = out.text;
  }
  return { carry: lastOut, finalText: isFinalRound ? lastOut : '' };
}

async function runLinearRound(mode, task, P, r, rounds, carry, isFinalRound, totals, tx) {
  let incoming = carry;
  let fromRole = carry ? `${P.agents[P.agents.length - 1].role} (prev round)` : null;
  let lastOut = '';

  for (let i = 0; i < P.agents.length && !cancelled; i++) {
    const a = P.agents[i];
    const isLastAgent = i === P.agents.length - 1;
    const decode = (mode === 'text') || (isFinalRound && isLastAgent);

    vizSetActive(a.key, { decode, state: decode ? 'decoding → text' : (mode === 'text' ? 'reasoning…' : 'latent…') });
    await sleep(120);

    const system = a.prompt;
    let user = buildUser(task, incoming, r, rounds, fromRole) + modeTail(mode, decode);

    const out = await runAgentCall({ a, system, user, decode, mode, totals, tx });
    if (cancelled) break;

    // Deliberation tool hop: if the tool-caller asked to search, run it & let it answer.
    if (P.tools && a.key === 'toolcaller') {
      const m = out.text.match(/SEARCH_WEB\(([^)]+)\)/i);
      if (m) {
        const q = m[1].trim();
        const toolMsg = addMsg(tx, '🔎', 'Tool · Wikipedia', 'tool', `searching: "${esc(q)}"…`);
        const result = await searchWeb(q);
        toolMsg.querySelector('.msg-body').innerHTML = `🔎 SEARCH_WEB("${esc(q)}") → ${esc(result.slice(0, 320))}`;
        // tool-caller now answers using the result
        vizSetActive(a.key, { decode, state: 'with tool result' });
        const u2 = `TASK: ${task}\nTool result for "${q}":\n${result}\n${modeTail(mode, decode)}`;
        const out2 = await runAgentCall({ a, system, user: u2, decode, mode, totals, tx, suffix: ' (post-tool)' });
        out.text = out2.text;
      }
    }

    incoming = out.text;
    fromRole = a.role;
    lastOut = out.text;
  }
  const finalText = isFinalRound ? lastOut : '';
  return { carry: lastOut, finalText };
}

async function runMixtureRound(mode, task, P, r, rounds, carry, isFinalRound, totals, tx) {
  const specialists = P.agents.filter(a => a.parallel);
  const summarizer  = P.agents.find(a => !a.parallel);
  const contribs = [];

  for (const a of specialists) {
    if (cancelled) break;
    vizSetActive(a.key, { state: mode === 'text' ? 'reasoning…' : 'latent…' });
    await sleep(120);
    const user = buildUser(task, carry, r, rounds, carry ? 'Summarizer (prev round)' : null) + modeTail(mode, false);
    const out = await runAgentCall({ a, system: a.prompt, user, decode: false, mode, totals, tx });
    contribs.push(`[${a.role}]: ${out.text}`);
  }
  if (cancelled) return { carry, finalText: '' };

  const realDecode = (mode === 'text') || isFinalRound; // summarizer decodes full text only on the final round
  vizSetActive(summarizer.key, { decode: realDecode, state: realDecode ? 'decoding → text' : 'latent…' });
  await sleep(120);
  const sUser = `TASK: ${task}\nRECURSION ROUND ${r} of ${rounds}.\nSpecialist contributions:\n${contribs.join('\n')}` +
                modeTail(mode, realDecode);
  const sOut = await runAgentCall({ a: summarizer, system: summarizer.prompt, user: sUser, decode: realDecode, mode, totals, tx });
  return { carry: sOut.text, finalText: isFinalRound ? sOut.text : '' };
}

// Compute the agent's REAL last-layer hidden state via the model's get_last_hidden
// graph function, run it through the RecursiveLink (if trained weights are present),
// and render proof + the latent-space routing similarity into the transcript message.
async function emitLatentProof(msg, system, user) {
  const line = el('div', 'latent-proof', '🧠 computing latent (get_last_hidden)…');
  msg.appendChild(line);

  if (!latentRT.ok) {
    line.innerHTML = `🧠 <span class="lp-warn">latent unavailable</span> · ${esc(latentRT.reason || '')} — using compressed-text thought`;
    return null;
  }

  const info = await latentForward(latentRT, `${system}\n\n${user}`);
  if (!info.ok) {
    line.innerHTML = `🧠 <span class="lp-warn">get_last_hidden failed</span> · ${esc(info.error || '')}`;
    return null;
  }

  let vec = info.vector, linkNote = '';
  if (vec && recursiveLinks?.links?.length) {
    try { vec = recursiveLinks.links[0].apply(vec); linkNote = ' · RecursiveLink R_out applied'; }
    catch { linkNote = ' · RecursiveLink skipped (dim mismatch)'; }
  } else if (vec) {
    linkNote = ' · RecursiveLink: identity (no trained weights yet)';
  }

  let cosNote = '';
  if (vec && prevLatentVec) {
    const c = cosine(prevLatentVec, vec);
    if (c != null) cosNote = ` · cos(prev)=${c.toFixed(3)}`;
  }
  if (vec) prevLatentVec = vec;

  const shapeStr = info.shape ? `[${info.shape.join('×')}]` : '?';
  const normStr  = info.norm != null ? ` · ‖h‖=${info.norm.toFixed(2)}` : '';
  const poolNote = info.vector ? '' : ` · ${esc(info.note || 'values unavailable (f16)')}`;
  line.innerHTML = `🧠 <span class="lp-ok">get_last_hidden</span> → ${esc(shapeStr)} ${esc(info.dtype || '')}` +
                   `${normStr}${cosNote}${esc(linkNote)}${poolNote}`;
  return info;
}

// One agent generation + transcript entry (final decode is streamed).
async function runAgentCall({ a, system, user, decode, mode, totals, tx, suffix = '' }) {
  const kind = decode ? 'decode' : (mode === 'text' ? 'text' : 'latent');
  const msg = addMsg(tx, a.emoji, a.role + suffix, kind, '', kind);
  const body = msg.querySelector('.msg-body');

  // Real latent space: for every intermediate (non-decode) step of the RecursiveMAS
  // run, push the agent's prompt through the model's get_last_hidden graph function
  // and surface proof of the on-device last-layer hidden state + latent-space routing.
  if (kind === 'latent' && latentRT) await emitLatentProof(msg, system, user);

  let out;
  if (decode) {
    body.textContent = '';
    out = await generate({ system, user, maxTokens: MAXTOK_DECODE, stream: true, onToken: (d) => { body.textContent += d; body.scrollIntoView({ block: 'nearest' }); } });
    if (!out.text) body.textContent = '(no output)';
  } else {
    body.textContent = '…';
    out = await generate({ system, user, maxTokens: mode === 'text' ? MAXTOK_TEXT : MAXTOK_LATENT });
    body.textContent = out.text || '(no output)';
  }

  totals.tokens += out.tokens;
  totals.time   += out.time;
  totals.calls  += 1;
  msg.querySelector('.msg-meta').textContent = `${out.tokens} tok · ${(out.time / 1000).toFixed(1)}s`;
  return out;
}

// ════════════════════════════════════════════════════════════════════════════
// Transcript rendering
// ════════════════════════════════════════════════════════════════════════════
function txContainer(mode) { return mode === 'text' ? tText : tLatent; }
function addRoundDivider(tx, r, rounds) {
  tx.appendChild(el('div', 'round-divider', `Recursion round ${r} of ${rounds}${r === rounds ? ' · final → decode to text' : ' · latent only'}`));
}
function addMsg(tx, emoji, role, kind, bodyHtml, badge) {
  const cls = kind === 'decode' ? 'msg decode-msg' : kind === 'latent' ? 'msg latent-msg' : kind === 'tool' ? 'msg tool-msg' : 'msg';
  const m = el('div', cls);
  const badgeHtml = badge ? `<span class="badge-mini badge-${badge}">${badge}</span>` : '';
  m.innerHTML =
    `<div class="msg-head"><span class="msg-role">${emoji} ${esc(role)}${badgeHtml}</span><span class="msg-meta"></span></div>` +
    `<div class="msg-body"></div>`;
  if (bodyHtml) m.querySelector('.msg-body').innerHTML = bodyHtml;
  tx.appendChild(m);
  return m;
}

// Small note in the transcript showing which prior latents an agent retrieved.
function addRetrievalNote(tx, a, hits) {
  const body = hits.length
    ? '🧲 retrieved from bus: ' + hits.map(h => `${h.emoji}${esc(h.role)}·r${h.round} <b>${h.score.toFixed(2)}</b>`).join(' · ')
    : '🧲 bus empty — nothing to retrieve yet';
  const m = el('div', 'retrieval-note', `${a.emoji} ${esc(a.role)} &nbsp;←&nbsp; ${body}`);
  tx.appendChild(m);
}

// Inline note for the latent-only run (skips, fallbacks, aborts).
function addChainNote(tx, text) {
  tx.appendChild(el('div', 'latent-proof', `🧬 <span class="lp-warn">${esc(text)}</span>`));
}

// Proof line for a latent-only step: the real hidden tensor / decode, injection, routing.
function addChainProof(msg, info) {
  const parts = [];
  if (info.decoded) {
    parts.push('<span class="lp-ok">decoded ← injected latent</span>');
    if (info.injected === false) parts.push('(latent injection skipped)');
    if (info.nTokens != null) parts.push(`${info.nTokens} tok`);
  } else {
    parts.push('<span class="lp-ok">get_last_hidden</span>');
    if (info.shape) parts.push(`→ [${info.shape.join('×')}] ${esc(info.dtype || '')}`);
    parts.push(info.injected ? 'latent injected (R_out=I)' : 'no prefix (first hop)');
    if (info.norm != null) parts.push(`‖h‖=${info.norm.toFixed(2)}`);
    if (info.cos != null) parts.push(`cos(prev)=${info.cos.toFixed(3)}`);
  }
  msg.appendChild(el('div', 'latent-proof', `🧬 ${parts.join(' · ')}`));
}

// ════════════════════════════════════════════════════════════════════════════
// Embedding memory bus visualization
// ════════════════════════════════════════════════════════════════════════════
function resetBus() {
  latentMemory = [];
  if (busList) busList.innerHTML = '';
  if (busDim) busDim.textContent = '';
}
function addBusEntry(entry, dim) {
  busCard.style.display = 'block';
  if (busDim && dim) busDim.textContent = `${latentMemory.length} vectors · ${dim}-dim`;
  const chip = el('div', 'bus-entry',
    `<span class="bus-entry-tag">${entry.emoji} ${esc(entry.role)} · r${entry.round}</span>` +
    `<span class="bus-entry-text">${esc(entry.text.slice(0, 90))}${entry.text.length > 90 ? '…' : ''}</span>`);
  chip.style.animation = 'busIn 0.4s ease';
  busList.appendChild(chip);
  busList.scrollTop = busList.scrollHeight;
}

// ════════════════════════════════════════════════════════════════════════════
// Metrics
// ════════════════════════════════════════════════════════════════════════════
function renderMetrics() {
  const L = results.latent, T = results.text;
  metricsCard.style.display = 'block';
  metricsGrid.innerHTML = '';
  const row = (label, lval, tval) => {
    metricsGrid.appendChild(el('div', 'mcell mlabel', label));
    metricsGrid.appendChild(el('div', 'mcell mval latent', lval));
    metricsGrid.appendChild(el('div', 'mcell mval text', tval));
  };
  const busLabel = L?.totals.embedCalls ? 'RecursiveMAS · 🧲 bus' : 'RecursiveMAS';
  metricsGrid.appendChild(el('div', 'mcell mhead', 'metric'));
  metricsGrid.appendChild(el('div', 'mcell mhead', busLabel));
  metricsGrid.appendChild(el('div', 'mcell mhead', 'Text-MAS'));
  row('Total tokens', L ? L.totals.tokens : '—', T ? T.totals.tokens : '—');
  row('End-to-end time', L ? (L.totals.time / 1000).toFixed(1) + ' s' : '—', T ? (T.totals.time / 1000).toFixed(1) + ' s' : '—');
  row('LLM calls', L ? L.totals.calls : '—', T ? T.totals.calls : '—');
  if (L?.totals.embedCalls) row('Embedding calls', L.totals.embedCalls, '—');

  metricsDelta.innerHTML = '';
  if (L && T) {
    const tokRed = (1 - L.totals.tokens / T.totals.tokens) * 100;
    const speedup = T.totals.time / L.totals.time;
    const pill = (val, label) => { const p = el('div', 'delta-pill'); p.innerHTML = `<div class="dval">${val}</div><div class="dlabel">${label}</div>`; return p; };
    metricsDelta.appendChild(pill(tokRed >= 0 ? `${tokRed.toFixed(1)}%` : '—', 'fewer tokens'));
    metricsDelta.appendChild(pill(speedup >= 1 ? `${speedup.toFixed(2)}×` : `${speedup.toFixed(2)}×`, 'speed-up'));
    metricsDelta.appendChild(pill(`${L.totals.calls} vs ${T.totals.calls}`, 'calls (latent vs text)'));
    const note = el('p', 'muted', '↑ The paper reports 34.6–75.6% fewer tokens and 1.2–2.4× speed-up; your numbers depend on the model, task, and rounds.');
    note.style.flexBasis = '100%';
    metricsDelta.appendChild(note);
  } else {
    metricsDelta.innerHTML = '<p class="muted">Run both modes to see the head-to-head reduction &amp; speed-up.</p>';
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Run orchestration
// ════════════════════════════════════════════════════════════════════════════
function setRunning(on) {
  running = on;
  [runLatentBtn, runTextBtn, runBothBtn].forEach(b => b.disabled = on);
  stopBtn.style.display = on ? 'inline-flex' : 'none';
}

function showTranscriptTab(mode) {
  transcriptCard.style.display = 'block';
  document.querySelectorAll('.ttab').forEach(t => t.classList.toggle('active', t.dataset.tab === mode));
  tLatent.style.display = mode === 'latent' ? 'flex' : 'none';
  tText.style.display   = mode === 'text' ? 'flex' : 'none';
}

async function run(modes) {
  if (running) return;
  const task = taskInput.value.trim();
  if (!task) { taskInput.focus(); return; }

  cancelled = false;
  busEnabled = !!busToggle?.checked;
  chainEnabled = !!$('chain-toggle')?.checked;
  setRunning(true);
  setModelStatus(busEnabled ? 'Preparing (chat + embedder)…' : 'Preparing…', 'loading');

  try {
    const ok = await ensureModel();
    if (!ok) { setRunning(false); return; }

    const pattern = getPattern();
    const rounds  = getRounds();
    buildLoopViz();
    resetBus();
    if (!busEnabled) busCard.style.display = 'none';

    setModelStatus('Running agents…', 'running');

    for (const mode of modes) {
      if (cancelled) break;
      const tx = txContainer(mode);
      tx.innerHTML = '';
      showTranscriptTab(mode);
      vizReset();
      const res = await runMode(mode, task, pattern, rounds, tx);
      results[mode] = res;
      renderMetrics();
    }

    vizReset();
    vizRound.textContent = cancelled ? 'stopped' : 'done';
    setModelStatus(cancelled ? 'Stopped' : `${MODELS.find(m => m.id === chatModelId)?.label} ready`, cancelled ? 'idle' : 'ready');
  } catch (e) {
    console.error('RecursiveMAS run failed:', e);
    vizRound.textContent = 'error';
    setModelStatus('Run failed: ' + (e?.message || e), 'error');
    modelHint.textContent = busEnabled
      ? 'The embedding bus loads a second model — if your GPU is low on memory, try unchecking the bus or a smaller backbone.'
      : 'See the browser console for details.';
  } finally {
    setRunning(false);
    loopViz.querySelector('.feedback-arrow')?.classList.remove('active');
  }
}

runLatentBtn.onclick = () => { results.text = null; run(['latent']); };
runTextBtn.onclick   = () => { results.latent = null; run(['text']); };
runBothBtn.onclick   = () => { run(['text', 'latent']); };  // baseline first, then RecursiveMAS
stopBtn.onclick      = () => { cancelled = true; setModelStatus('Stopping…', 'loading'); };

const chainToggle = $('chain-toggle');
function refreshLatentBtnLabel() {
  const tag = chainEnabled ? '🧬 latent-only' : busEnabled ? '🧲 bus' : 'latent';
  runLatentBtn.innerHTML = `▶ Run RecursiveMAS <span class="tag tag-latent">${tag}</span>`;
}
busToggle?.addEventListener('change', () => {
  busEnabled = busToggle.checked;
  // The embedding bus and vector-only transfer are mutually exclusive routing schemes.
  if (busEnabled && chainEnabled && chainToggle) { chainToggle.checked = false; chainEnabled = false; }
  refreshLatentBtnLabel();
});
chainToggle?.addEventListener('change', () => {
  chainEnabled = chainToggle.checked;
  if (chainEnabled && busEnabled && busToggle) { busToggle.checked = false; busEnabled = false; }
  refreshLatentBtnLabel();
});

document.querySelectorAll('.ttab').forEach(t => t.onclick = () => showTranscriptTab(t.dataset.tab));

// Append any custom (self-compiled) models to the backbone picker.
function renderCustomModelCards() {
  const wrap = $('model-cards');
  if (!wrap || !CUSTOM_MODELS.length) return;
  CUSTOM_MODELS.forEach(c => {
    const label = el('label', 'model-card');
    const isChecked = c.model_id === 'recursivemas-0.5b' ? 'checked' : '';
    label.innerHTML =
      `<input type="radio" name="model" value="${esc(c.model_id)}" ${isChecked} />` +
      `<div class="model-card-inner">` +
      `<div class="model-name">${esc(c.label || c.model_id)}${c.exposesLatent ? ' <span class="tag tag-latent">latent</span>' : ''}</div>` +
      `<div class="model-meta">${esc(c.size || 'custom')}</div></div>`;
    wrap.appendChild(label);
  });
}

// ── Boot ──────────────────────────────────────────────────────────────────────
renderCustomModelCards();
renderPattern();
$('model-cards')?.addEventListener('change', maybeLoadLink);
maybeLoadLink();
checkWebGPU().then(ok => {
  if (!ok) setModelStatus('WebGPU not detected — see hint below', 'error');
});
