/* ════════════════════════════════════════════════════════════════════════════
   RecursiveLink — in-browser forward of the paper's inner/outer link.

     inner:  R_in(h)  = h     + W2·GELU(W1·h)
     outer:  R_out(h) = W3·h  + W2·GELU(W1·h)

   Loads weights produced offline by model-build/train_recursivelink.py
   (recursivelink.json). The matmuls are tiny (hidden ≈ 512–1024, bottleneck ≈ 256)
   so plain JS is fine — this is NOT the expensive part.

   The expensive/blocked part is getting `h` (last-layer hidden state) out of the
   model and feeding R(h) back in — see LOW_LEVEL_NOTES at the bottom.
   ════════════════════════════════════════════════════════════════════════════ */

function gelu(x) {                              // tanh approximation (matches export "gelu":"tanh")
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

// y = W·x + b ;  W is [out][in] (row-major), x is [in], b is [out] (optional)
function matVec(W, x, b) {
  const out = new Float32Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = b ? b[i] : 0;
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

export class RecursiveLink {
  /** @param {{w1,b1,w2,b2,w3?}} weights one link's matrices (arrays) */
  constructor(weights) { this.w = weights; }

  /** Apply to a single hidden vector h (Float32Array | number[]) -> Float32Array. */
  apply(h) {
    const { w1, b1, w2, b2, w3, scale } = this.w;
    // Unit-normalize first (matches training: a pooled hidden is ~100s× an embedding,
    // so direction is learned in normalized space and `scale` restores magnitude).
    let n = 0; for (let i = 0; i < h.length; i++) n += h[i] * h[i];
    n = Math.sqrt(n) + 1e-6;
    const hn = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) hn[i] = h[i] / n;
    const z = matVec(w1, hn, b1);              // bottleneck
    for (let i = 0; i < z.length; i++) z[i] = gelu(z[i]);
    const proj = matVec(w2, z, b2);            // -> target dim
    const res = w3 ? matVec(w3, hn) : hn;      // outer: W3·hn ; inner: identity
    const s = (scale == null ? 1 : scale);     // learned embedding magnitude
    const out = new Float32Array(proj.length);
    for (let i = 0; i < out.length; i++) out[i] = (res[i] + proj[i]) * s;
    return out;
  }

  /** Apply across a sequence of hidden vectors [seq][hidden]. */
  applySeq(hs) { return hs.map(h => this.apply(h)); }
}

/** Load recursivelink.json -> { hidden, links: RecursiveLink[] }. */
export async function loadRecursiveLinks(url) {
  const j = await (await fetch(url)).json();
  return { hidden: j.hidden, links: j.links.map(l => new RecursiveLink(l)) };
}

/* ──────────────────────────────────────────────────────────────────────────────
   LOW_LEVEL_NOTES — wiring true latent transfer (the remaining research piece)

   With a model compiled via the recursiveMASWebLLM repo (exposing get_last_hidden),
   the loop is:

     1. tokens -> input embeds:        vm.getFunction("embed")(inputIds)
     2. last-layer hidden states:      [hidden, kv] = vm.getFunction("get_last_hidden")(embeds, kv)
     3. transfer in latent space:      next = outerLink.applySeq(hidden)   // or innerLink for recurrence
     4. feed back WITHOUT decoding:    vm.getFunction("prefill"/"get_last_hidden")(next, kv)
     5. only the final round decodes:  logits = vm.getFunction("prefill")(next, kv); sample

   WebLLM's high-level engine.chat.completions does NOT expose arbitrary functions.
   You reach the TVM module via the engine's internal pipeline (tvmjs runtime):
   load the wasm with tvmjs, grab `vm.getFunction(name)`, and manage the paged KV
   cache yourself (create_paged_kv_cache + the cache object threaded through calls).
   This is below the documented API and is the part to prototype next.
   ────────────────────────────────────────────────────────────────────────────── */
export const LOW_LEVEL_NOTES = 'See the recursiveMASWebLLM repo and the comment above.';
