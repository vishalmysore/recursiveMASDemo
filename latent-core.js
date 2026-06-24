/* ════════════════════════════════════════════════════════════════════════════
   latent-core.js — make the APP actually use latent space.

   Off-the-shelf WebLLM only gives you input_ids → logits (text). Our custom model
   (RecursiveMAS-0.5B) was compiled with an extra graph function `get_last_hidden`
   that returns the LAST-LAYER HIDDEN STATES (the paper's "latent thoughts") with no
   LM head applied. This module reaches WebLLM's main-thread pipeline (created via
   CreateMLCEngine) and invokes that function on-device, so each agent emits a REAL
   latent vector instead of only a compressed-text proxy.

   Trick: WebLLM's own pipeline.embedAndForward() already does the entire KV-cache
   plumbing (embed → begin_forward → prefill(emb, kv, params) → end_forward). Since
   our `get_last_hidden` has the *identical* (input_embed, kv_cache, params) signature
   as `prefill`, we temporarily swap pipeline.prefill → get_last_hidden and reuse all
   of WebLLM's correct plumbing, then restore. Everything is wrapped: any failure
   returns { ok:false, reason } so the app degrades gracefully to its text behaviour.
   ════════════════════════════════════════════════════════════════════════════ */

// Decode one IEEE-754 half (f16 bit pattern) → JS number. Inverse of latent-chain's
// f32to16; needed because tvmjs NDArray.toArray() throws "Unsupported data type
// float16", but toRawBytes() gives us the raw halves to decode ourselves.
export function f16to32(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0)    return s * f * 5.9604644775390625e-8;   // subnormal: f * 2^-24
  if (e === 0x1f) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);        // normal
}

// Read a CPU-resident NDArray into a Float32Array, transparently decoding float16
// (which toArray() cannot handle) via raw bytes. Returns null if unreadable.
export function ndToFloat32(cpu) {
  const dt = (cpu?.dtype || '').toLowerCase();
  if (dt.includes('float16') || dt === 'f16') {
    try {
      const bytes = cpu.toRawBytes();                                  // Uint8Array, 2 bytes/half
      const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
      const out = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) out[i] = f16to32(u16[i]);
      return out;
    } catch { return null; }
  }
  try { return cpu.toArray(); } catch { return null; }
}

// Resolve the low-level latent runtime for a loaded model, or explain why it can't.
export function getLatentRuntime(engine, modelId) {
  try {
    const pipeline = engine?.loadedModelIdToPipeline?.get?.(modelId);
    if (!pipeline) return { ok: false, reason: `no pipeline loaded for "${modelId}"` };
    const vm = pipeline.vm;
    if (!vm?.getFunction) return { ok: false, reason: 'pipeline has no tvm VM (worker engine?)' };

    // tvmjs requires an active memory scope around getFunction (it returns a TVM
    // object); detach the handles so they outlive the scope — exactly how web-llm
    // itself fetches prefill/decode. Without beginScope, getFunction throws
    // "Must call beginScope to use functions that returns TVM objects", which the
    // old bare catch silently mislabeled as "function not in the model lib".
    let fGLH = null, fDLH = null;
    const tvm = pipeline.tvm;
    tvm.beginScope();
    try {
      try { fGLH = tvm.detachFromCurrentScope(vm.getFunction('get_last_hidden')); } catch { /* not in lib */ }
      try { fDLH = tvm.detachFromCurrentScope(vm.getFunction('decode_last_hidden')); } catch { /* optional */ }
    } finally {
      tvm.endScope();
    }
    if (!fGLH) {
      return { ok: false, reason: 'get_last_hidden is not in this model lib — it is a stock model, not the latent-exposing build' };
    }
    if (typeof pipeline.embedAndForward !== 'function' || typeof pipeline.resetChat !== 'function') {
      return { ok: false, reason: 'incompatible web-llm version (no embedAndForward/resetChat)' };
    }
    // Capture the REAL forward handles up-front (latentForward swaps pipeline.prefill
    // temporarily; the latent-chain decode loop needs the genuine LM-head prefill/decode).
    return {
      ok: true, pipeline, vm, tvm, fGLH, fDLH,
      realPrefill: pipeline.prefill,
      realDecode:  pipeline.decoding,
      stopTokens:  Array.isArray(pipeline.stopTokens) ? pipeline.stopTokens.slice() : [],
    };
  } catch (e) {
    return { ok: false, reason: e?.message || String(e) };
  }
}

// Run a single forward through the model and return the REAL last-layer hidden state
// for `text`, pooled (mean over sequence) into one vector. Proof of on-device latent use.
export async function latentForward(rt, text) {
  if (!rt?.ok) return { ok: false, error: rt?.reason || 'no latent runtime' };
  const { pipeline, tvm, fGLH, fDLH } = rt;

  let tokens;
  try {
    const enc = pipeline.tokenizer.encode((text || ' ').slice(0, 6000) || ' ');
    tokens = Array.from(enc);
  } catch (e) { return { ok: false, error: 'tokenize: ' + (e?.message || e) }; }
  if (!tokens.length) return { ok: false, error: 'empty token sequence' };

  // Keep within one prefill chunk (this is a single forward, not chunked prefill).
  const cap = (pipeline.prefillChunkSize || 2048) - 1;
  if (tokens.length > cap) tokens = tokens.slice(-cap);

  const savedPrefill = pipeline.prefill;
  const savedDecode  = pipeline.decoding;
  let shape = null, dtype = null, vector = null, norm = null, seq = null, dim = null;
  let forwardErr = null, note = null;

  // embedAndForward and tvm.empty both return TVM objects, so the whole forward+pool
  // must run inside ONE tvm memory scope (mirrors web-llm's own prefill loop). We read
  // plain JS values out inside the scope, so nothing needs detaching — endScope frees
  // the GPU hidden tensor and the CPU copy automatically (no leak across agent calls).
  tvm.beginScope();
  try {
    pipeline.resetChat(true);          // clear KV cache + re-add sequence 0 (keep stats)
    pipeline.prefill = fGLH;           // identical (emb, kv, params) signature
    if (fDLH) pipeline.decoding = fDLH;
    const hiddenGPU = await pipeline.embedAndForward([tokens], tokens.length);

    // hiddenGPU: NDArray [1, seq, hidden] (model dtype, typically f16). Pool on CPU.
    shape = (hiddenGPU.shape || []).slice();
    dtype = hiddenGPU.dtype;
    dim = shape[shape.length - 1] || null;
    seq = shape.length >= 2 ? shape[shape.length - 2] : 1;
    const cpu = tvm.empty(shape, dtype, tvm.cpu());
    cpu.copyFrom(hiddenGPU);
    await pipeline.device.sync();
    const flat = ndToFloat32(cpu);   // toArray, or decode f16 raw bytes
    if (flat && dim && flat.length >= seq * dim) {
      vector = new Float32Array(dim);
      for (let t = 0; t < seq; t++) {
        const off = t * dim;
        for (let d = 0; d < dim; d++) vector[d] += flat[off + d];
      }
      let s = 0;
      for (let d = 0; d < dim; d++) { vector[d] /= seq; s += vector[d] * vector[d]; }
      norm = Math.sqrt(s);
    }
  } catch (e) {
    if (shape == null) forwardErr = 'forward: ' + (e?.message || e);  // failed before a tensor came back
    else note = 'pool: ' + (e?.message || e);                         // got the tensor, only pooling failed
  } finally {
    tvm.endScope();
    pipeline.prefill = savedPrefill; pipeline.decoding = savedDecode;
    try { pipeline.resetChat(true); } catch { /* leave clean */ }
  }

  if (forwardErr) return { ok: false, error: forwardErr };
  // A numerically unstable build (e.g. f16 overflow in the residual stream) returns
  // NaN/Inf hidden states. Surface that as a clear failure instead of passing a poisoned
  // latent downstream, where it becomes NaN logits and aborts the wasm runtime.
  if (vector && !Number.isFinite(norm)) {
    return { ok: false, error: 'model returned non-finite hidden states (NaN/Inf) — numerically unstable build', stage: 'nan' };
  }
  if (note)       return { ok: true, shape, dtype, seq, dim, vector: null, norm: null, note };
  return { ok: true, shape, dtype, seq, dim, vector, norm };
}

// Cosine similarity between two pooled latent vectors (latent-space routing signal).
export function cosine(a, b) {
  if (!a || !b || a.length !== b.length) return null;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den ? dot / den : null;
}
