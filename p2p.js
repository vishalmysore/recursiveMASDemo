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
import { chainForward, chainDecode, encodeLatentSeq, decodeLatentSeq } from './latent-chain.js';
import { loadRecursiveLinks } from './recursive-link.js';
import { PeerManager } from './peer-manager.js';

// ── The one shared model (must match on both peers) ──────────────────────────────
const MODEL_ID  = 'recursivemas-0.5b';
const MODEL_LBL = 'RecursiveMAS-0.5B';
const CUSTOM_MODEL = {
  model:    'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC',
  model_id: MODEL_ID,
  // ?v=N cache-buster — bump whenever the .wasm is rebuilt (WebLLM caches by URL).
  model_lib:'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC/resolve/main/libs/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm?v=2',
  vram_required_MB: 900,
  // Trained RecursiveLink — maps the latent sequence into the model's input-embedding
  // space so the vector-only decode is coherent. Produced by recursiveMASWebLLM's
  // train_recursivelink.py. NOTE: GitHub Release assets are NOT CORS-enabled, so the
  // browser can't fetch them — host on Hugging Face (same repo as the model, CORS-OK).
  // If it fails to load, the inject paths fall back to the raw latent.
  recursiveLink: 'https://huggingface.co/VishalMysore/RecursiveMAS-0.5B-MLC/resolve/main/recursivelink.json',
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
// When someone joins via an invite link, default them to the role that pairs
// with the host's (Travel ⇄ Hotel) so the two agents actually differ.
const ROLE_COMPLEMENT = { travel: 'hotel', hotel: 'travel', custom: 'custom' };

const TEMP = 0.6, MAX_DECODE = 320;

// ── DOM ──────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
// Compact SDP codec: deflate-raw → URL-safe base64, so an offer fits inside a
// shareable #hash link (a raw SDP base64 is ~3× longer). Both peers need WebGPU,
// which implies a browser with Compression/DecompressionStream.
const toB64u   = (bytes) => btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const fromB64u = (str) => { const s = str.trim().replace(/-/g, '+').replace(/_/g, '/'); return Uint8Array.from(atob(s + '='.repeat((4 - s.length % 4) % 4)), c => c.charCodeAt(0)); };
async function encodeSDP(desc) {
  const json = JSON.stringify({ type: desc.type, sdp: desc.sdp });
  const cs = new CompressionStream('deflate-raw');
  const w = cs.writable.getWriter(); w.write(new TextEncoder().encode(json)); w.close();
  return toB64u(new Uint8Array(await new Response(cs.readable).arrayBuffer()));
}
async function decodeSDP(token) {
  const ds = new DecompressionStream('deflate-raw');
  const w = ds.writable.getWriter(); w.write(fromB64u(token)); w.close();
  return JSON.parse(new TextDecoder().decode(await new Response(ds.readable).arrayBuffer()));
}
const waLink = (text) => 'https://wa.me/?text=' + encodeURIComponent(text);
function wireShare(btn, hint, text) {
  btn.onclick = () => {
    navigator.clipboard?.writeText(text).catch(() => {});
    hint.style.display = 'inline';
    setTimeout(() => { hint.style.display = 'none'; }, 2200);
  };
}

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
  if (info.decoded) {
    p.push(`<span class="lp-ok">${info.injected ? 'decoded ← shared latent (vector-only)' : 'decoded (prompt only — latent not injected)'}</span>`);
    if (info.nTokens != null) p.push(`${info.nTokens} tok`);
  }
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

// The model must be loaded before either peer can start the WebRTC handshake,
// so gate both "start connecting" buttons on it and explain why when disabled.
function updateConnectBtns() {
  const ci = $('create-invite-btn'), jn = $('join-btn');
  const cih = $('invite-gate-hint'), jnh = $('join-gate-hint');
  if (ci) ci.disabled = !myReady;
  if (jn) jn.disabled = !myReady;
  if (cih) cih.style.display = myReady ? 'none' : '';
  if (jnh) jnh.style.display = myReady ? 'none' : '';
}

function updateRunBtn() {
  const btn = $('run-btn'), hint = $('run-hint');
  const connected = !!(pm && pm.getConnected().length > 0);
  const ok = isHost && connected && myReady && peerReady;
  btn.disabled = !ok;
  // Spell out exactly what's still blocking the run, most-blocking first.
  let msg, warn = true;
  if (!myReady)         msg = '⚠️ Load the RecursiveMAS-0.5B model first — use the “⤓ Load” button in step 1 above.';
  else if (!connected)  msg = isHost ? 'Create an invite link above and connect your peer to enable the run.'
                                     : 'Connect to the host above to join the collaboration.';
  else if (!peerReady)  msg = '⏳ Waiting for the other agent to finish loading its model…';
  else if (!isHost)     msg = '✓ Connected. The host drives the run — ask them to click ▶.';
  else { msg = '✓ Both agents loaded and connected — click ▶ to run.'; warn = false; }
  if (hint) { hint.textContent = msg; hint.classList.toggle('run-hint-warn', warn); }
  btn.title = ok ? '' : msg;
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
  // Load the trained RecursiveLink if one is published — it maps the latent into the
  // model's input-embedding space so the vector-only decode is coherent. Optional: if
  // it's absent or fails to load, the inject paths fall back to magnitude calibration.
  let linkNote = '';
  if (CUSTOM_MODEL.recursiveLink) {
    try {
      const { links } = await loadRecursiveLinks(CUSTOM_MODEL.recursiveLink);
      rt.link = links?.[0] || null;
      if (rt.link) linkNote = ' + RecursiveLink';
    } catch (e) { console.warn('[p2p] RecursiveLink load failed, using fallback:', e?.message || e); }
  }
  myReady = true;
  setStatus(`Model ready · latent ✓${linkNote} — you can connect & collaborate`, 'ready');
  if (pm) pm.broadcast({ type: 'ready' });
  updateConnectBtns();
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
      showConnected(name);
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

let pendingOffer = null; // set when the page is opened from an invite link

function showConnected(name) {
  $('conn-ok-text').textContent = `Connected to ${name} — ready to collaborate below.`;
  $('conn-ok').style.display = 'flex';
  $('connect-host').style.display = 'none';
  $('connect-join').style.display = 'none';
}

// Host: build a one-tap invite link (offer + role baked into the #hash).
async function createInvite() {
  isHost = true;
  const btn = $('create-invite-btn'); btn.disabled = true; btn.textContent = 'Creating…';
  try {
    const offer = await ensurePM().createOffer('peer');
    const params = new URLSearchParams();
    params.set('offer', await encodeSDP(offer));
    params.set('role', myRoleKey());
    const url = `${location.origin}${location.pathname}#${params.toString()}`;
    $('invite-url').value = url;
    $('invite-ready').style.display = '';
    wireShare($('copy-invite-btn'), $('invite-copied'), url);
    $('wa-share').href = waLink(`Join me for a RecursiveMAS latent-space collab — open this in Chrome/Edge:\n${url}`);
    addWire('invite link ready — share it, then paste your friend\'s answer code.');
  } catch (e) { addNote('invite error: ' + (e?.message || e), true); }
  finally { btn.disabled = false; btn.textContent = '🔗 Create invite link'; updateRunBtn(); }
}

// Joiner: the offer came in via the link; produce the answer code to send back.
async function joinWithOffer() {
  isHost = false;
  if (!pendingOffer) { addNote('No invite found in this link.', true); return; }
  const btn = $('join-btn'); btn.disabled = true; btn.textContent = 'Generating…';
  try {
    const answer = await ensurePM().acceptOffer('host', pendingOffer);
    const code = await encodeSDP(answer);
    $('answer-out').value = code;
    $('answer-ready').style.display = '';
    wireShare($('copy-answer-btn'), $('answer-copied'), code);
    $('wa-answer').href = waLink(`Here's my RecursiveMAS answer code — paste it into your Connect box:\n\n${code}`);
    addWire('answer code generated — send it back to the host.');
  } catch (e) {
    addNote('join error: ' + (e?.message || e), true);
    btn.disabled = false; btn.textContent = 'Generate my answer code';
  }
}

// Host: friend's answer code arrives → finish the handshake.
async function acceptAnswer() {
  const raw = $('answer-in').value.trim();
  if (!raw) return;
  const btn = $('accept-answer-btn'); btn.disabled = true;
  try {
    await ensurePM().setAnswer('peer', await decodeSDP(raw));
    addWire('answer accepted — establishing connection…');
  } catch (e) { addNote('connect error: ' + (e?.message || e), true); btn.disabled = false; }
}

// On load: if there's an invite in the #hash, flip into joiner mode and default
// to the role that pairs with the host's.
async function initFromHash() {
  const params = new URLSearchParams(location.hash.replace(/^#/, ''));
  const offerToken = params.get('offer');
  if (!offerToken) return;
  try {
    pendingOffer = await decodeSDP(offerToken);
  } catch { addNote('This invite link looks invalid or corrupted.', true); return; }
  const hostRole = params.get('role') || 'travel';
  const myRole = ROLE_COMPLEMENT[hostRole] || 'travel';
  const radio = document.querySelector(`input[name=role][value="${myRole}"]`);
  if (radio) { radio.checked = true; syncRolePrompt(); }
  $('connect-host').style.display = 'none';
  $('connect-join').style.display = '';
  $('invited-role-hint').textContent =
    `Host is the ${ROLE_NAME[hostRole]} agent, so you're set up as the ${ROLE_NAME[myRole]} agent (change above if you like). Load the model, then generate your code.`;
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

async function handleHop(from, { seq, b64, rows, dim, task, rounds }) {
  if (busy) return; busy = true;
  try {
    const c = seq + 1;                       // this is the c-th computation in the chain
    const isFinal = c === 2 * rounds;
    const prefix = { data: await decodeLatentSeq(b64, rows, dim), rows };   // full latent sequence
    // Role only — NO task text. The task entered latent space once (the host's first
    // forward encoded it); from then on it travels solely as the latent sequence. Re-feeding
    // the task as text here would mean the answer comes from the prompt, not the latent.
    const prompt = `${myPrompt()}\n\nBuild on the incoming latent state from the other agent.`;
    addWire(`◀ received latent from ${from} (hop ${seq}/${2 * rounds}) — ${rows}×${dim} sequence`);

    if (isFinal) {
      const msg = addMsg('🧩', `${ROLE_NAME[myRoleKey()]} · decoding shared latent`, 'decode');
      const body = msg.querySelector('.msg-body');
      // True to the RecursiveMAS premise: the final answer is decoded FROM the shared
      // latent sequence (inject = true), with no task text in the prompt — this is the
      // single place a latent becomes text. Nothing on the wire was ever text.
      const userMsg = 'Decode the team\'s shared latent state into the final answer.';
      const res = await chainDecode(rt, prompt, prefix, {
        inject: true, system: myPrompt(), user: userMsg,
        maxTokens: MAX_DECODE, temperature: TEMP, onToken: d => { body.textContent += d; },
      });
      if (!res.ok) { addNote(`latent decode failed (${res.stage}): ${res.error}`, true); pm.broadcast({ type: 'note', text: 'decode failed' }); return; }
      if (!body.textContent) body.textContent = res.text || '(no output)';
      addProof(msg, { decoded: true, injected: res.injected, nTokens: res.nTokens });
      pm.broadcast({ type: 'final', text: res.text, fromRole: ROLE_NAME[myRoleKey()] });
    } else {
      const msg = addMsg('🧬', `${ROLE_NAME[myRoleKey()]} · latent only`, 'latent');
      msg.querySelector('.msg-body').textContent = '⟶ latent only (no text decode)';
      const res = await chainForward(rt, prompt, prefix);
      if (!res.ok || !res.data) { addNote(`latent forward failed (${res.stage || 'pool'}): ${res.error || 'no vector'}`, true); pm.broadcast({ type: 'note', text: 'latent forward failed' }); return; }
      addProof(msg, { shape: res.shape, dtype: res.dtype, norm: res.norm, injected: res.injected, cos: prevVec ? cosine(prevVec, res.pooled) : null });
      prevVec = res.pooled;
      const enc = await encodeLatentSeq(res.data, res.rows, res.dim);
      pm.broadcast({ type: 'latent-hop', seq: c, b64: enc.b64, rows: enc.rows, dim: enc.dim, task, rounds });
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
    if (!res.ok || !res.data) { addNote(`latent forward failed (${res.stage || 'pool'}): ${res.error || 'no vector'}`, true); return; }
    addProof(msg, { shape: res.shape, dtype: res.dtype, norm: res.norm, injected: false });
    prevVec = res.pooled;
    addWire(`▶ sending latent to peer (hop 1/${2 * rounds})`);
    const enc = await encodeLatentSeq(res.data, res.rows, res.dim);
    pm.broadcast({ type: 'latent-hop', seq: 1, b64: enc.b64, rows: enc.rows, dim: enc.dim, task, rounds });
  } catch (e) { addNote('run error: ' + (e?.message || e), true); }
  finally { busy = false; }
}

// ── Wire up UI ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('input[name=role]').forEach(r => r.addEventListener('change', syncRolePrompt));
$('load-btn').onclick          = loadModel;
$('create-invite-btn').onclick = createInvite;
$('join-btn').onclick          = joinWithOffer;
$('accept-answer-btn').onclick = acceptAnswer;
$('be-host-btn').onclick       = () => { location.hash = ''; location.reload(); };
$('run-btn').onclick           = run;
syncRolePrompt();
initFromHash();
updateConnectBtns();
updateRunBtn();
if (!navigator.gpu) setStatus('WebGPU not detected — open in Chrome/Edge 113+', 'error');
