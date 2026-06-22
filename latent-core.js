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

// Resolve the low-level latent runtime for a loaded model, or explain why it can't.
export function getLatentRuntime(engine, modelId) {
  try {
    const pipeline = engine?.loadedModelIdToPipeline?.get?.(modelId);
    if (!pipeline) return { ok: false, reason: `no pipeline loaded for "${modelId}"` };
    const vm = pipeline.vm;
    if (!vm?.getFunction) return { ok: false, reason: 'pipeline has no tvm VM (worker engine?)' };

    let fGLH = null, fDLH = null;
    try { fGLH = vm.getFunction('get_last_hidden'); } catch { /* not in lib */ }
    try { fDLH = vm.getFunction('decode_last_hidden'); } catch { /* optional */ }
    if (!fGLH) {
      return { ok: false, reason: 'get_last_hidden is not in this model lib — it is a stock model, not the latent-exposing build' };
    }
    if (typeof pipeline.embedAndForward !== 'function' || typeof pipeline.resetChat !== 'function') {
      return { ok: false, reason: 'incompatible web-llm version (no embedAndForward/resetChat)' };
    }
    // Capture the REAL forward handles up-front (latentForward swaps pipeline.prefill
    // temporarily; the latent-chain decode loop needs the genuine LM-head prefill/decode).
    return {
      ok: true, pipeline, vm, tvm: pipeline.tvm, fGLH, fDLH,
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
  let hiddenGPU = null;
  try {
    pipeline.resetChat(true);          // clear KV cache + re-add sequence 0 (keep stats)
    pipeline.prefill = fGLH;           // identical (emb, kv, params) signature
    if (fDLH) pipeline.decoding = fDLH;
    hiddenGPU = await pipeline.embedAndForward([tokens], tokens.length);
  } catch (e) {
    pipeline.prefill = savedPrefill; pipeline.decoding = savedDecode;
    try { pipeline.resetChat(true); } catch { /* ignore */ }
    return { ok: false, error: 'forward: ' + (e?.message || e) };
  }
  pipeline.prefill = savedPrefill; pipeline.decoding = savedDecode;

  // hiddenGPU: NDArray [1, seq, hidden] (model dtype, typically f16). Pool on CPU.
  let shape = null, dtype = null, vector = null, norm = null, seq = null, dim = null;
  try {
    shape = (hiddenGPU.shape || []).slice();
    dtype = hiddenGPU.dtype;
    dim = shape[shape.length - 1] || null;
    seq = shape.length >= 2 ? shape[shape.length - 2] : 1;
    const cpu = tvm.empty(shape, dtype, tvm.cpu());
    cpu.copyFrom(hiddenGPU);
    await pipeline.device.sync();
    let flat = null;
    try { flat = cpu.toArray(); } catch { /* f16 toArray may be unsupported; shape is still proof */ }
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
    // We still got a real hidden tensor; just couldn't pool it. Report shape anyway.
    return { ok: true, shape, dtype, seq, dim, vector: null, norm: null, note: 'pool: ' + (e?.message || e) };
  } finally {
    try { pipeline.resetChat(true); } catch { /* leave clean */ }
  }
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
