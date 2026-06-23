/* ════════════════════════════════════════════════════════════════════════════
   latent-chain.js — TRUE vector-only RecursiveMAS transfer.

   The paper's mechanism: intermediate agents NEVER decode text. Each agent forwards
   over [ R_out(previous latent) ⊕ its own prompt embeddings ] through the model and
   emits a last-layer hidden state (latent thought); that latent — not text — is what
   the next agent consumes. Only the final agent in the final round decodes to text.

   How this module realises that on top of WebLLM 0.2.78 (main-thread engine):

     • chainForward()  — intermediate agents. Builds input embeddings as a single
       injected latent token (the previous agent's pooled hidden, R_out = identity
       since both spaces are 896-d for the same model) prepended to the agent's prompt
       token embeddings, runs the custom `get_last_hidden` graph fn, pools the result
       to one 896-d latent vector. NO sampling, NO text.

     • chainDecode()   — final agent. Same injected-latent seed, but uses the REAL
       prefill (LM head → logits) and then an autoregressive sampling loop reusing
       WebLLM's own sampleTokenFromLogits / getTokensEmbeddings / decode handles.

   Latent state crosses agents as a CPU Float32 vector (no GPU tensor kept alive across
   calls) and is re-injected as one f16 token, so there is no tensor-lifetime bookkeeping
   and the per-hop sequence stays bounded. Every stage is guarded; a failure is returned
   as { ok:false, error, stage } so the caller can fall back to the text path.
   ════════════════════════════════════════════════════════════════════════════ */

import { ndToFloat32, f16to32 } from './latent-core.js';

// ── Full-sequence latent transport ────────────────────────────────────────────────
// We send the WHOLE last-hidden sequence (not a mean-pooled vector) so the latent
// carries per-token detail and the trained RecursiveLink can't collapse to a constant.
// Pack it compactly so it fits in one WebRTC message: keep the last MAX_LATENT_ROWS
// positions, f32 → f16 → deflate-raw → base64 (≈6× smaller than a JSON float array).
const MAX_LATENT_ROWS = 64;
const _b64 = (bytes) => { let s = ''; const C = 8192; for (let i = 0; i < bytes.length; i += C) s += String.fromCharCode.apply(null, bytes.subarray(i, i + C)); return btoa(s); };
const _unb64 = (b64) => { const s = atob(b64); const a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; };

export async function encodeLatentSeq(flat, rows, dim) {
  let r = rows, off = 0;
  if (rows > MAX_LATENT_ROWS) { off = (rows - MAX_LATENT_ROWS) * dim; r = MAX_LATENT_ROWS; }
  const n = r * dim;
  const u16 = new Uint16Array(n);
  for (let i = 0; i < n; i++) u16[i] = f32to16(flat[off + i]);
  const cs = new CompressionStream('deflate-raw');
  const w = cs.writable.getWriter(); w.write(new Uint8Array(u16.buffer)); w.close();
  const buf = new Uint8Array(await new Response(cs.readable).arrayBuffer());
  return { b64: _b64(buf), rows: r, dim };
}
export async function decodeLatentSeq(b64, rows, dim) {
  const ds = new DecompressionStream('deflate-raw');
  const w = ds.writable.getWriter(); w.write(_unb64(b64)); w.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(rows * dim);
  for (let i = 0; i < out.length; i++) out[i] = f16to32(u16[i]);
  return out;
}

// ── float32 → float16 (round-to-nearest-even), for uploading the latent token ──
const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);
function f32to16(val) {
  _f32[0] = val;
  const x = _u32[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = (x >>> 23) & 0xff;
  let mant = x & 0x007fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);     // Inf / NaN
  exp = exp - 127 + 15;
  if (exp >= 0x1f) return sign | 0x7c00;                           // overflow → Inf
  if (exp <= 0) {                                                  // subnormal / zero
    if (exp < -10) return sign;
    mant |= 0x00800000;
    const shift = 14 - exp;
    let half = mant >> shift;
    if ((mant >> (shift - 1)) & 1) half += 1;                      // round
    return sign | half;
  }
  let half = sign | (exp << 10) | (mant >> 13);
  if (mant & 0x1000) half += 1;                                    // round-to-nearest-even
  return half;
}

const isF16 = (dt) => typeof dt === 'string' && (dt === 'float16' || dt === 'f16' || dt.includes('16'));

// Build a [1, dim] embedding tensor on-device from a pooled CPU latent vector.
function latentToken(rt, vec, dtype) {
  const { tvm, pipeline } = rt;
  const dim = vec.length;
  if (isF16(dtype)) {
    const u16 = new Uint16Array(dim);
    for (let i = 0; i < dim; i++) u16[i] = f32to16(vec[i]);
    // tvmjs copyFrom rejects float16 (no JS Float16Array) → "Unsupported data type
    // float16". Upload the raw half-float bytes instead, which has no dtype check.
    const nd = tvm.empty([1, dim], 'float16', pipeline.device);
    nd.copyFromRawBytes(new Uint8Array(u16.buffer));
    return nd;
  }
  return tvm.empty([1, dim], dtype || 'float32', pipeline.device).copyFrom(vec);
}

// Build a [rows, dim] embedding tensor on-device from a flat CPU latent sequence.
function latentTokens(rt, flat, rows, dtype) {
  const { tvm, pipeline } = rt;
  const dim = flat.length / rows;
  if (isF16(dtype)) {
    const u16 = new Uint16Array(rows * dim);
    for (let i = 0; i < rows * dim; i++) u16[i] = f32to16(flat[i]);
    const nd = tvm.empty([rows, dim], 'float16', pipeline.device);
    nd.copyFromRawBytes(new Uint8Array(u16.buffer));
    return nd;
  }
  return tvm.empty([rows, dim], dtype || 'float32', pipeline.device).copyFrom(flat);
}

// Map an incoming latent sequence through the trained RecursiveLink (R_out), per
// position. Without a link, returns the raw sequence (out-of-distribution — the
// untrained case). flat is [rows*dim]; returns a new [rows*dim] Float32Array.
function applyLinkSeq(rt, flat, rows, dim) {
  if (!(rt.link && typeof rt.link.apply === 'function')) return flat;
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) out.set(rt.link.apply(flat.subarray(r * dim, (r + 1) * dim)), r * dim);
  return out;
}

// Accept either a full sequence { data, rows } or a bare pooled vector (back-compat
// with main.js's single-browser demo) and normalize to { data, rows }.
function asSeq(prefix, dim) {
  if (!prefix) return null;
  if (prefix.data && prefix.rows) return prefix.data.length === prefix.rows * dim ? prefix : null;
  const data = prefix instanceof Float32Array ? prefix : (Array.isArray(prefix) ? Float32Array.from(prefix) : null);
  if (!data || data.length === 0 || data.length % dim !== 0) return null;
  return { data, rows: data.length / dim };
}

// Scale a pooled latent so its L2 norm matches the mean per-token norm of the prompt's
// input embeddings. A pooled last-hidden vector has a far larger magnitude than a normal
// input embedding, so injecting it raw is wildly out-of-distribution and derails the
// model. Best effort: returns the raw vector unchanged if anything can't be measured.
async function calibrateLatentNorm(rt, vec, tokEmb, dim, seqLen) {
  try {
    const { tvm, pipeline } = rt;
    const cpu = tvm.empty(tokEmb.shape, tokEmb.dtype, tvm.cpu());
    cpu.copyFrom(tokEmb);
    await pipeline.device.sync();
    const flat = ndToFloat32(cpu);
    const seqN = tokEmb.shape[0] || seqLen || 1;
    if (!flat || flat.length < seqN * dim) return vec;
    let embNorm = 0;
    for (let t = 0; t < seqN; t++) {
      let s = 0; const o = t * dim;
      for (let k = 0; k < dim; k++) { const v = flat[o + k]; s += v * v; }
      embNorm += Math.sqrt(s);
    }
    embNorm /= seqN;
    let latNorm = 0;
    for (let k = 0; k < dim; k++) latNorm += vec[k] * vec[k];
    latNorm = Math.sqrt(latNorm);
    if (embNorm > 0 && latNorm > 0) return Float32Array.from(vec, x => x * (embNorm / latNorm));
    return vec;
  } catch { return vec; }
}

// Mean-pool a CPU-resident hidden tensor [.. , seq, dim] → Float32Array[dim] + L2 norm.
function poolHidden(flat, seq, dim) {
  if (!flat || flat.length < seq * dim) return { vec: null, norm: null };
  const vec = new Float32Array(dim);
  for (let t = 0; t < seq; t++) {
    const off = t * dim;
    for (let k = 0; k < dim; k++) vec[k] += flat[off + k];
  }
  let s = 0;
  for (let k = 0; k < dim; k++) { vec[k] /= seq; s += vec[k] * vec[k]; }
  return { vec, norm: Math.sqrt(s) };
}

// Common KV plumbing: begin_forward → fn(emb, kv, params) → end_forward. Mirrors
// WebLLM's embedAndForward. Returns the raw tuple (caller reads .get(0)).
function rawForward(rt, fn, allEmb, total) {
  const { pipeline, tvm } = rt;
  pipeline.fKVCacheBeginForward(pipeline.kvCache, tvm.makeShapeTuple([0]), tvm.makeShapeTuple([total]));
  const ret = fn(allEmb, pipeline.kvCache, pipeline.params);
  pipeline.fKVCacheEndForward(pipeline.kvCache);
  pipeline.filledKVCacheLength += total;
  return ret;
}

// Tokenize + cap to one prefill chunk.
function encodeCapped(rt, text) {
  const ids = Array.from(rt.pipeline.tokenizer.encode((text || ' ').slice(0, 6000) || ' '));
  const cap = (rt.pipeline.prefillChunkSize || 2048) - 2;
  return ids.length > cap ? ids.slice(-cap) : ids;
}

// Format (system, user) with the MODEL'S OWN chat template (e.g. ChatML for the
// Qwen-based RecursiveMAS build) by reusing web-llm's loaded Conversation object.
// Without this the decoder is fed a raw concatenated string with no role markers,
// which makes a small instruct model ramble (often in Chinese). Returns the fully
// templated prompt string ending in the assistant reply header, or null if the
// template can't be reached — caller then falls back to the raw text. The shared
// conversation is reset before and after, so engine state is left clean.
export function formatChatPrompt(rt, system, user) {
  const conv = rt?.pipeline?.conversation;
  if (!conv || typeof conv.getPromptArray !== 'function' || !conv.config?.roles) return null;
  try {
    conv.reset();
    if (system != null && system !== '') conv.override_system_message = system;
    conv.appendMessage('user', String(user ?? ''));
    conv.appendReplyHeader('assistant');           // model continues as the assistant
    const arr = conv.getPromptArray();             // text-only model → array of strings
    return arr.map(x => (Array.isArray(x) ? x.join('') : x)).join('');
  } catch { return null; }
  finally { try { conv.reset(); } catch { /* leave clean */ } }
}

// ── Intermediate agent: latent in → latent out, never decodes text ─────────────
// `prefix` is the incoming latent SEQUENCE { data: Float32Array[rows*dim], rows } or null.
export async function chainForward(rt, text, prefix) {
  if (!rt?.ok) return { ok: false, error: rt?.reason || 'no runtime', stage: 'runtime' };
  const { pipeline, tvm, fGLH } = rt;
  let tokens;
  try { tokens = encodeCapped(rt, text); } catch (e) { return { ok: false, error: e.message, stage: 'tokenize' }; }
  if (!tokens.length) return { ok: false, error: 'empty tokens', stage: 'tokenize' };

  try {
    pipeline.resetChat(true);
    tvm.beginScope();
    const tokEmb = pipeline.getTokensEmbeddings(tokens);          // [seq, dim]
    const dim = tokEmb.shape[tokEmb.shape.length - 1];
    let all = tokEmb, total = tokens.length, injected = false;
    const pseq = asSeq(prefix, dim);
    if (pseq) {
      try {
        // Map the incoming latent SEQUENCE into input-embedding space with the trained
        // RecursiveLink (R_out), per position; prepend the whole sequence to the prompt.
        const linked = applyLinkSeq(rt, pseq.data, pseq.rows, dim);
        const pre = latentTokens(rt, linked, pseq.rows, tokEmb.dtype);   // [rows, dim]
        all = tvm.concatEmbeddings([pre, tokEmb]); total = tokens.length + pseq.rows; injected = true;
      } catch (e) { all = tokEmb; total = tokens.length; console.warn('[chain] latent inject failed:', e?.message || e); }  // forward prompt only
    }
    all = all.view([1].concat(all.shape));                        // [1, total, dim]
    const ret = rawForward(rt, fGLH, all, total);
    const hidden = ret.get ? ret.get(0) : ret;                   // [1, total, dim]
    const shape = (hidden.shape || []).slice();
    const dtype = hidden.dtype;
    const seq = shape[shape.length - 2] || total, d = shape[shape.length - 1] || dim;
    const cpu = tvm.empty(shape, dtype, tvm.cpu());
    cpu.copyFrom(hidden);
    await pipeline.device.sync();
    const flat = ndToFloat32(cpu);   // full [seq*dim] hidden sequence
    tvm.endScope();
    const { vec, norm } = poolHidden(flat, seq, d);   // pooled only for the proof badge
    return { ok: true, shape, dtype, dim: d, seq, rows: seq, data: flat, pooled: vec, norm, injected };
  } catch (e) {
    try { tvm.endScope(); } catch { /* ignore */ }
    return { ok: false, error: e?.message || String(e), stage: 'forward' };
  } finally {
    try { pipeline.resetChat(true); } catch { /* leave clean */ }
  }
}

// ── Final agent: seed with the accumulated latent SEQUENCE, then decode text ───
// `prefix` is the incoming latent sequence { data: Float32Array[rows*dim], rows } or null.
export async function chainDecode(rt, text, prefix, opts = {}) {
  if (!rt?.ok) return { ok: false, error: rt?.reason || 'no runtime', stage: 'runtime' };
  const { pipeline, tvm, realPrefill, realDecode } = rt;
  const stopTokens = (rt.stopTokens && rt.stopTokens.length ? rt.stopTokens : pipeline.stopTokens) || [];
  const maxTokens = opts.maxTokens || 256;
  // Repetition penalty is essential here. web-llm's sampler damps repeats by reading
  // pipeline.appearedTokensFreq — but that map is maintained by the NORMAL decode loop,
  // not ours. Without feeding it (see below) a small model collapses into degenerate
  // "35 35 35 …" loops. We pass an explicit penalty and keep the frequency map updated.
  const genConfig = {
    temperature: opts.temperature ?? 0.6,
    top_p: opts.top_p ?? 0.95,
    repetition_penalty: opts.repetition_penalty ?? 1.3,
  };
  // Prefer the model's real chat template (keeps the final answer coherent); fall
  // back to the raw text if a system/user split wasn't supplied or the template is
  // unreachable.
  const promptText = (opts.system != null || opts.user != null)
    ? (formatChatPrompt(rt, opts.system, opts.user) ?? text)
    : text;
  let tokens;
  try { tokens = encodeCapped(rt, promptText); } catch (e) { return { ok: false, error: e.message, stage: 'tokenize' }; }

  const out = [];
  let dim = null, logits = null, injected = false, emitted = '';
  try {
    pipeline.resetChat(true);
    // 1. Seed forward through the REAL prefill (LM head → logits).
    tvm.beginScope();
    const tokEmb = pipeline.getTokensEmbeddings(tokens);
    dim = tokEmb.shape[tokEmb.shape.length - 1];
    let all = tokEmb, total = tokens.length;
    // Seed the decoder with the shared latent SEQUENCE (opts.inject) so the answer is
    // decoded FROM the latent, not the prompt. The trained RecursiveLink (R_out) maps
    // each position into input-embedding space; without a link the raw sequence is
    // injected (the out-of-distribution, untrained case).
    const pseq = opts.inject ? asSeq(prefix, dim) : null;
    if (pseq) {
      try {
        const linked = applyLinkSeq(rt, pseq.data, pseq.rows, dim);
        const pre = latentTokens(rt, linked, pseq.rows, tokEmb.dtype);
        all = tvm.concatEmbeddings([pre, tokEmb]); total = tokens.length + pseq.rows; injected = true;
      } catch (e) { all = tokEmb; total = tokens.length; console.warn('[chain] decode inject failed:', e?.message || e); }
    }
    all = all.view([1].concat(all.shape));
    const ret = rawForward(rt, realPrefill, all, total);
    logits = tvm.detachFromCurrentScope(ret.get ? ret.get(0) : ret);
    tvm.endScope();

    // Reset the per-generation token-frequency map the sampler penalises against.
    try { pipeline.appearedTokensFreq?.clear?.(); } catch { /* ignore */ }

    // 2. Autoregressive sampling loop (reuses WebLLM's sampler).
    for (let step = 0; step < maxTokens; step++) {
      if (opts.isStopped && opts.isStopped()) break;
      const tok = await pipeline.sampleTokenFromLogits(logits, genConfig);
      try { logits.dispose && logits.dispose(); } catch { /* ignore */ }
      if (stopTokens.includes(tok)) break;
      out.push(tok);
      // Feed the sampler's repetition-penalty bookkeeping (the normal pipeline loop
      // does this); without it appearedTokensFreq stays empty and penalty is a no-op.
      try { const f = pipeline.appearedTokensFreq?.get?.(tok); pipeline.appearedTokensFreq?.set?.(tok, (f || 0) + 1); } catch { /* ignore */ }
      try {
        const full = pipeline.tokenizer.decode(Int32Array.from(out));
        const piece = full.slice(emitted.length); emitted = full;
        if (piece && opts.onToken) opts.onToken(piece);
      } catch { /* decode-on-the-fly best effort */ }
      tvm.beginScope();
      const e1 = pipeline.getTokensEmbeddings([tok]).view([1, 1, dim]);
      const ret2 = rawForward(rt, realDecode, e1, 1);
      logits = tvm.detachFromCurrentScope(ret2.get ? ret2.get(0) : ret2);
      tvm.endScope();
    }
    try { logits && logits.dispose && logits.dispose(); } catch { /* ignore */ }
    let textOut = emitted;
    if (!textOut) { try { textOut = pipeline.tokenizer.decode(Int32Array.from(out)); } catch { /* ignore */ } }
    return { ok: true, text: (textOut || '').trim(), nTokens: out.length, injected };
  } catch (e) {
    try { tvm.endScope(); } catch { /* ignore */ }
    return { ok: false, error: e?.message || String(e), stage: 'decode', partial: emitted };
  } finally {
    try { pipeline.resetChat(true); } catch { /* leave clean */ }
  }
}
