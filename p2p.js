/* ════════════════════════════════════════════════════════════════════════════
   p2p.js — RecursiveMAS over WebRTC.

   Two browsers, each running the SAME custom RecursiveMAS-0.5B model with a different
   role prompt. The host drives N latent rounds: agents alternate, each forwarding
   through get_last_hidden over the OTHER agent's injected latent (a 896-d vector sent
   across the WebRTC data channel) and never decoding text — until the final hop, where
   one agent decodes the shared latent into the answer. Reuses latent-chain.js for the
   on-device latent maths and peer-manager.js for transport.
   ════════════════════════════════════════════════════════════════════════════ */
import * as webllm from '@mlc-ai/web-llm';
import { getLatentRuntime, cosine } from './latent-core.js';
import { chainForward, chainDecode } from './latent-chain.js';
import { PeerManager } from './peer-manager.js';

// ── The one shared model (must match on both peers) ──────────────────────────────
const MODEL_ID  = 'recursivemas-0.5b';
const MODEL_LBL = 'RecursiveMAS-0.5B';
const CUSTOM_MODEL = {
  model:    'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC',
  model_id: MODEL_ID,
  model_lib:'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC/resolve/main/libs/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm',
  vram_required_MB: 900,
};
const appConfig = () => ({
  ...webllm.prebuiltAppConfig,
  model_list: [...webllm.prebuiltAppConfig.model_list, CUSTOM_MODEL],
});

const ROLE_PROMPTS = {
  travel: 'You are a Travel Planner agent. You design concise day-by-day itineraries — routes, transport, timing, and what to see. You collaborate with a Hotel agent who handles lodging. Be specific and brief.',
  hotel:  'You are a Hotel Concierge agent. You recommend specific lodging by neighbourhood, price band, and proximity to the day plan. You collaborate with a Travel agent. Be specific and brief.',
  custom: '',
};
const ROLE_NAME = { travel: 'Travel', hotel: 'Hotel', custom: 'Agent' };

const TEMP = 0.6, MAX_DECODE = 320;

// ── DOM ──────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const enc = (obj) => btoa(JSON.stringify(obj));
const dec = (str) => JSON.parse(atob(str.trim()));

const tx = $('transcript');
function addNote(text, warn) { tx.appendChild(el('div', 'latent-proof', `🛰️ <span class="${warn?'lp-warn':'lp-ok'}">${esc(text)}</span>`)); tx.scrollTop = tx.scrollHeight; }
function addWire(text) { tx.appendChild(el('div', 'wire-msg', `📡 ${esc(text)}`)); tx.scrollTop = tx.scrollHeight; }
function addMsg(emoji, role, kind) {
  const m = el('div', kind === 'decode' ? 'msg decode-msg' : 'msg latent-msg');
  m.innerHTML = `<div class="msg-head"><span class="msg-role">${esc(emoji)} ${esc(role)}</span><span class="msg-meta"></span></div><div class="msg-body"></div>`;
  tx.appendChild(m); tx.scrollTop = tx.scrollHeight; return m;
}
function addProof(msg, info) {
  const p = [];
  if (info.decoded) { p.push('<span class="lp-ok">decoded ← injected latent</span>'); if (info.nTokens != null) p.push(`${info.nTokens} tok`); }
  else {
    p.push('<span class="lp-ok">get_last_hidden</span>');
    if (info.shape) p.push(`→ [${info.shape.join('×')}] ${esc(info.dtype || '')}`);
    p.push(info.injected ? 'latent injected (R_out=I)' : 'no prefix (first hop)');
    if (info.norm != null) p.push(`‖h‖=${info.norm.toFixed(2)}`);
    if (info.cos != null) p.push(`cos(prev)=${info.cos.toFixed(3)}`);
  }
  msg.appendChild(el('div', 'latent-proof', `🧬 ${p.join(' · ')}`));
}

// ── State ──────────────────────────────────────────────────────────────────────────
let engine = null, rt = null, myReady = false;
let pm = null, isHost = false, peerReady = false, prevVec = null, busy = false;

function myRoleKey() { return document.querySelector('input[name=role]:checked')?.value || 'travel'; }
function myPrompt()  { return $('role-prompt').value.trim() || ROLE_PROMPTS[myRoleKey()] || ROLE_PROMPTS.travel; }
function myName()    { return $('agent-name').value.trim() || `${ROLE_NAME[myRoleKey()]}-${Math.random().toString(36).slice(2,5)}`; }

function setStatus(text, cls) { const b = $('model-status'); b.textContent = cls; b.className = 'status-badge status-' + cls; $('model-hint').textContent = text; }
function syncRolePrompt() { const k = myRoleKey(); $('role-prompt').value = ROLE_PROMPTS[k] || ''; if (!$('agent-name').value) $('agent-name').placeholder = `${ROLE_NAME[k]}-agent`; }

function updateRunBtn() {
  const btn = $('run-btn');
  const ok = isHost && pm && pm.getConnected().length > 0 && myReady && peerReady;
  btn.disabled = !ok;
  btn.title = ok ? '' : 'Needs: you = host, peer connected, both models loaded';
}

// ── Model load ─────────────────────────────────────────────────────────────────────
async function loadModel() {
  if (engine) return true;
  if (!navigator.gpu) { setStatus('WebGPU not available — use Chrome/Edge 113+', 'error'); return false; }
  setStatus('Loading RecursiveMAS-0.5B (one-time download)…', 'loading');
  try {
    engine = await webllm.CreateMLCEngine(MODEL_ID, {
      appConfig: appConfig(),
      initProgressCallback: (p) => setStatus(p.text || `Loading… ${Math.round((p.progress||0)*100)}%`, 'loading'),
    });
  } catch (e) { setStatus('Load failed: ' + (e?.message || e), 'error'); return false; }
  rt = getLatentRuntime(engine, MODEL_ID);
  if (!rt.ok) { setStatus('Latent runtime unavailable: ' + rt.reason, 'error'); return false; }
  myReady = true;
  setStatus('Model ready · latent ✓ — you can connect & collaborate', 'ready');
  if (pm) pm.broadcast({ type: 'ready' });
  updateRunBtn();
  return true;
}

// ── Peer manager / signalling ───────────────────────────────────────────────────────
function ensurePM() {
  if (pm) return pm;
  pm = new PeerManager({
    myName: myName(), myPersona: myPrompt(), myModelLabel: MODEL_LBL,
    onPeerJoin: (name, hello) => {
      addWire(`connected to ${name} — role: ${String(hello.persona || '').slice(0, 48)}…`);
      renderPeers();
      if (myReady) pm.broadcast({ type: 'ready' });
      updateRunBtn();
    },
    onPeerLeave: (name) => { addWire(`${name} disconnected`); peerReady = false; renderPeers(); updateRunBtn(); },
    onPeerState: () => renderPeers(),
    onMessage: (from, msg) => onMessage(from, msg),
  });
  return pm;
}

function renderPeers() {
  const wrap = $('peers'); wrap.innerHTML = '';
  if (!pm) return;
  for (const p of pm.peers.values()) {
    const cls = p.state === 'connected' ? 'on' : (p.state === 'connecting' ? 'wait' : '');
    wrap.appendChild(el('span', 'peer-pill', `<span class="dot ${cls}"></span> ${esc(p.name)} · ${esc(p.modelLabel || '…')}`));
  }
}

async function createInvite() {
  isHost = true;
  const offer = await ensurePM().createOffer('peer');
  $('offer-out').value = enc(offer);
  addWire('invite created — send the code to your friend, then paste their answer.');
  updateRunBtn();
}
async function joinWithOffer() {
  isHost = false;
  const offer = dec($('offer-in').value);
  const answer = await ensurePM().acceptOffer('host', offer);
  $('answer-out').value = enc(answer);
  addWire('answer made — send this code back to the host.');
}
async function acceptAnswer() {
  await ensurePM().setAnswer('peer', dec($('answer-in').value));
  addWire('answer accepted — establishing connection…');
}

// ── The latent protocol ──────────────────────────────────────────────────────────────
function onMessage(from, msg) {
  if (msg.type === 'ready') { peerReady = true; addWire(`${from} model ready`); updateRunBtn(); return; }
  if (msg.type === 'task')  { addNote(`task from ${from}: “${msg.task}” · ${msg.rounds} latent round(s)`); return; }
  if (msg.type === 'note')  { addNote(`${from}: ${msg.text}`, true); return; }
  if (msg.type === 'final') { addFinal(msg.fromRole || from, msg.text); return; }
  if (msg.type === 'latent-hop') { handleHop(from, msg); return; }
}

function addFinal(role, text) {
  const m = addMsg('🏁', `${role} · final answer`, 'decode');
  m.querySelector('.msg-body').textContent = text || '(no output)';
}

async function handleHop(from, { seq, vec, task, rounds }) {
  if (busy) return; busy = true;
  try {
    const c = seq + 1;                       // this is the c-th computation in the chain
    const isFinal = c === 2 * rounds;
    const prefix = Float32Array.from(vec);
    const prompt = `${myPrompt()}\n\nTASK: ${task}\nYou are mid-collaboration; build on the incoming latent state from the other agent.`;
    addWire(`◀ received latent from ${from} (hop ${seq}/${2 * rounds}) — ${vec.length}-d vector`);

    if (isFinal) {
      const msg = addMsg('🧩', `${ROLE_NAME[myRoleKey()]} · decoding`, 'decode');
      const body = msg.querySelector('.msg-body');
      const res = await chainDecode(rt, prompt, prefix, { maxTokens: MAX_DECODE, temperature: TEMP, onToken: d => { body.textContent += d; } });
      if (!res.ok) { addNote(`decode failed (${res.stage}): ${res.error}`, true); pm.broadcast({ type: 'note', text: 'decode failed' }); return; }
      if (!body.textContent) body.textContent = res.text || '(no output)';
      addProof(msg, { decoded: true, nTokens: res.nTokens });
      pm.broadcast({ type: 'final', text: res.text, fromRole: ROLE_NAME[myRoleKey()] });
    } else {
      const msg = addMsg('🧬', `${ROLE_NAME[myRoleKey()]} · latent only`, 'latent');
      msg.querySelector('.msg-body').textContent = '⟶ latent only (no text decode)';
      const res = await chainForward(rt, prompt, prefix);
      if (!res.ok || !res.pooled) { addNote(`latent forward failed (${res.stage || 'pool'}): ${res.error || 'no vector'}`, true); pm.broadcast({ type: 'note', text: 'latent forward failed' }); return; }
      addProof(msg, { shape: res.shape, dtype: res.dtype, norm: res.norm, injected: res.injected, cos: prevVec ? cosine(prevVec, res.pooled) : null });
      prevVec = res.pooled;
      pm.broadcast({ type: 'latent-hop', seq: c, vec: Array.from(res.pooled), task, rounds });
    }
  } catch (e) { addNote('hop error: ' + (e?.message || e), true); }
  finally { busy = false; }
}

async function run() {
  if (busy) return; busy = true;
  try {
    const task = $('task').value.trim();
    const rounds = parseInt($('rounds').value, 10) || 1;
    if (!task) return;
    tx.innerHTML = '';
    prevVec = null;
    addNote(`you (${ROLE_NAME[myRoleKey()]}) start · “${task}” · ${rounds} latent round(s) · final decode by the other agent`);
    pm.broadcast({ type: 'task', task, rounds });

    // c = 1: host produces the first latent (no incoming prefix), never decodes.
    const msg = addMsg('🧬', `${ROLE_NAME[myRoleKey()]} · latent only`, 'latent');
    msg.querySelector('.msg-body').textContent = '⟶ latent only (no text decode)';
    const res = await chainForward(rt, `${myPrompt()}\n\nTASK: ${task}`, null);
    if (!res.ok || !res.pooled) { addNote(`latent forward failed (${res.stage || 'pool'}): ${res.error || 'no vector'}`, true); return; }
    addProof(msg, { shape: res.shape, dtype: res.dtype, norm: res.norm, injected: false });
    prevVec = res.pooled;
    addWire(`▶ sending latent to peer (hop 1/${2 * rounds})`);
    pm.broadcast({ type: 'latent-hop', seq: 1, vec: Array.from(res.pooled), task, rounds });
  } catch (e) { addNote('run error: ' + (e?.message || e), true); }
  finally { busy = false; }
}

// ── Wire up UI ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('input[name=role]').forEach(r => r.addEventListener('change', syncRolePrompt));
$('load-btn').onclick          = loadModel;
$('create-invite-btn').onclick = () => createInvite().catch(e => addNote('invite error: ' + e.message, true));
$('join-btn').onclick          = () => joinWithOffer().catch(e => addNote('join error: ' + e.message, true));
$('accept-answer-btn').onclick = () => acceptAnswer().catch(e => addNote('connect error: ' + e.message, true));
$('run-btn').onclick           = run;
syncRolePrompt();
if (!navigator.gpu) setStatus('WebGPU not detected — open in Chrome/Edge 113+', 'error');
