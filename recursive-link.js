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

// Decode one IEEE-754 half (f16 bit pattern) → JS number.
function f16to32(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
  if (e === 0) return s * f * 5.9604644775390625e-8;
  if (e === 0x1f) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
}
// base64 of little-endian f16 bytes → Float32Array.
function f16b64(b64) {
  const s = atob(b64), bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = f16to32(u16[i]);
  return out;
}
const reshape = (flat, rows, cols) => { const W = new Array(rows); for (let r = 0; r < rows; r++) W[r] = flat.subarray(r * cols, (r + 1) * cols); return W; };

/** Load recursivelink.json (compact base64-f16 or legacy float arrays)
 *  -> { hidden, links: RecursiveLink[] }. */
export async function loadRecursiveLinks(url) {
  const j = await (await fetch(url)).json();
  const H = j.hidden, B = j.bottleneck;
  const dec = (l) => j.fmt === 'f16b64'
    ? { w1: reshape(f16b64(l.w1), B, H), b1: f16b64(l.b1),
        w2: reshape(f16b64(l.w2), H, B), b2: f16b64(l.b2),
        w3: l.w3 ? reshape(f16b64(l.w3), H, H) : undefined, scale: l.scale }
    : l;   // legacy: w1/b1/w2/b2 already float arrays
  return { hidden: H, links: j.links.map(l => new RecursiveLink(dec(l))) };
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
