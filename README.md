# ♻️ RecursiveMAS Playground

An interactive, **in-browser** demo of the paper [**“Recursive Multi-Agent Systems”**](https://recursivemas.github.io) (Yang, Zou, et al.) — built in the same spirit as [AgentHerd](https://vishalmysore.github.io/agentHerd/): real local LLMs running on **WebLLM + WebGPU**, no servers, no API keys, no installs, almost zero cost.

The paper's big idea: stop treating a multi-agent system as a chat transcript and start treating it as a **single recursive computation**. Agents are chained into a loop — `A₁ → A₂ → … → Aₙ → back to A₁` — and the whole loop is re-run for several *recursion rounds*. The intermediate rounds never decode to text; they collaborate in compact latent space. **Only the last agent in the last round speaks.** The result, versus ordinary text-passing multi-agent systems: **+8.3% accuracy, 1.2–2.4× faster, 34.6–75.6% fewer tokens.**

This playground lets you *feel* that on your own machine.

---

## 🚀 What it does

- **Runs the recursive agent loop with real local models.** Pick a backbone model (Qwen, Llama, Gemma, Phi), a collaboration pattern, and a recursion depth — then watch the loop execute live.
- **Compares RecursiveMAS against the text-passing baseline, head-to-head.** It runs both and shows you the token count, wall-clock time, and call count for each — reproducing the paper's Table 2 on your own hardware.
- **Animates the loop.** Active agent, inner/outer RecursiveLinks, the round counter, and the feedback edge that closes the loop are all visualized as it runs.
- **Implements the paper's four collaboration patterns.**

---

## 🧩 The four collaboration patterns

| Pattern | Agents | Idea |
|--------|--------|------|
| 🔗 **Sequential** | Planner → Critic → Solver | Chain-of-agents: decompose, judge, refine, solve. |
| 🧩 **Mixture** | Math · Code · Science → Summarizer | Domain specialists reason in parallel; a summarizer aggregates. |
| 🎓 **Distillation** | Expert → Learner | A larger expert distills reasoning to a smaller, faster learner. |
| 🛠️ **Deliberation** | Reflector ↔ Tool-Caller | Inner thinking paired with a real tool (live Wikipedia search). |

Each role shows the heterogeneous model the paper assigns to it (e.g. Planner → Qwen3-1.7B). In this demo one backbone model plays every role via a role-specific prompt, so it stays runnable on a single GPU.

---

## ⚙️ How the latent vs. text mechanism is demonstrated

| | Text-MAS (baseline) | RecursiveMAS (latent) |
|---|---|---|
| Intermediate agents | decode **full reasoning text**, passed wholesale downstream | emit a tiny **latent thought** (~25 words of dense state) |
| Final agent, final round | decodes text | decodes the full answer |
| Token cost | grows with every agent × every round | stays flat — only one full decode |

This is the paper's core efficiency mechanism: avoid repeatedly decoding intermediate agents to the vocabulary space, and pass a compact latent state instead.

> ⚠️ **Honest note.** Browser LLMs (WebLLM) do **not** expose model hidden states, so true latent-vector transfer and the *training* of the RecursiveLink modules can't run client-side. This demo runs real models and faithfully reproduces the **system behavior** — the recursive loop, the round structure, and the “only the final agent decodes” efficiency mechanism — by compressing the carried *text* in place of true latent vectors. The architecture, formulas, and training pipeline from the paper are explained in the “How it works” section in-app.

---

## 🧠 The RecursiveLink (explained in-app)

A lightweight two-layer residual module — the only thing the paper trains; the agents themselves stay frozen.

```
Inner link (within one agent):   R_in(h)  = h     + W₂·GELU(W₁·h)
Outer link (across agents):      R_out(h) = W₃·h  + W₂·GELU(W₁·h)
```

- **Inner link** feeds an agent's last-layer latent thought back as its own next input — deepening reasoning without decoding to text.
- **Outer link** projects one model's latent state into the next (heterogeneous) model's embedding space — the bridge that lets different model families collaborate.

Training is two-stage: an **inner loop** warm-starts each agent's inner link (cosine-similarity objective), then an **outer loop** unrolls the whole system over recursion rounds and back-propagates one shared cross-entropy signal through every outer link.

---

## 🧲 Embedding memory bus (optional) — passing *real* embeddings between agents

The honest limit above is that the "latent thought" passed between agents is compressed *text*. Tick **“Embedding memory bus”** to route state through actual vectors instead:

- A real on-device embedding model (`snowflake-arctic-embed-s`, ~33M) loads alongside the chat model.
- Each agent embeds its latent thought into a **shared in-browser vector store** (384-dim).
- The next agent **retrieves the top-k most similar prior latents by cosine similarity** instead of receiving the linear hand-off — you can watch the similarity scores in the transcript and the store fill up live.

This is the closest pure-browser analogue of the paper's **outer link**: state flows between agents as embeddings, addressed by similarity. The one caveat that can't be removed in a browser: what finally enters the chat LLM is still the *retrieved text* (a chat model can only ingest tokens, not a vector) — so the embeddings do the routing, not a true residual-stream transfer.

> **Why not a literally faithful latent transfer?** A capability spike confirmed that off-the-shelf in-browser model runtimes (both WebLLM and transformers.js) expose only `input_ids → logits`; they neither emit last-layer hidden states nor accept `inputs_embeds`. True vector-to-vector passing would require a custom-**compiled** model (offline) plus a trained RecursiveLink. The demo is wired to load such a model the moment you have one — see below.

---

## 🛠️ Using your own compiled model (latent-transfer-ready)

Unlike sealed ONNX exports, WebLLM models are **compiled by you** from an editable MLC-LLM (TVM) definition — and MLC already feeds *embeddings* (not token ids) into `prefill`. So a self-compiled model can expose last-layer hidden states and accept a latent vector back in, which is what a truly faithful RecursiveMAS needs. This app is already plumbed to use one.

The model is built in a **separate repo** — [**recursiveMASWebLLM**](https://github.com/vishalmysore/recursiveMASWebLLM) — which compiles the `.wasm` on GitHub Actions (no GPU needed: `mlc_llm compile` is codegen) and publishes the `.wasm` to a Release + the weights to Hugging Face.

**To use it here — one edit.** Uncomment the `CUSTOM_MODELS` record in [main.js](main.js) and point it at those URLs; it's merged into WebLLM's `appConfig` and auto-added to the backbone picker:

```js
const CUSTOM_MODELS = [{
  model:    'https://huggingface.co/vishalmysore/RecursiveMAS-0.5B-MLC',
  model_id: 'recursivemas-0.5b',
  model_lib:'https://github.com/vishalmysore/recursiveMASWebLLM/releases/download/model-RecursiveMAS-0.5B/RecursiveMAS-0.5B-q4f16_1-webgpu.wasm',
  recursiveLink: 'https://github.com/vishalmysore/recursiveMASWebLLM/releases/download/model-RecursiveMAS-0.5B/recursivelink.json',
  vram_required_MB: 900, label: 'RecursiveMAS 0.5B', size: '~0.5 GB · custom', exposesLatent: true,
}];
```

End users just open the app URL; WebLLM downloads the model from those URLs on first visit and caches it — identical to the built-in models, just self-hosted.

**Gotchas:** the `.wasm` must be compiled against a TVM/MLC version compatible with the pinned `@mlc-ai/web-llm`; hosts need permissive CORS (HF/GitHub are fine); and the page needs cross-origin isolation (COOP/COEP) — set those headers on your host, or use `coi-serviceworker` on GitHub Pages.

> Loading + running a custom model as a backbone works through this wiring today. Actually *reading its hidden states and looping them* still needs low-level JS calls into the TVM runtime (below the `chat.completions` API) — that's the remaining research piece, lives in [recursive-link.js](recursive-link.js), not the release step.

---

## 🖥️ Tech Stack

| What | Why |
|------|-----|
| **WebLLM** | Runs LLMs in the browser via WebGPU — no API key, no cloud |
| **WebGPU** | On-device inference |
| **Vite** | Build tooling and dev server |
| **GitHub Pages** | Static hosting — no server-side compute |

**Requirements:** A WebGPU browser (Chrome/Edge 113+). Integrated GPUs handle the 1–1.5B models; a discrete GPU is recommended for 3B+.

---

## 🏃 Local Development

```bash
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001). The first run downloads the chosen model (cached in your browser afterwards).

---

## 📦 Deployment

```bash
npm run deploy   # vite build && gh-pages -d dist
```

Set the GitHub Pages source to **GitHub Actions** (or the `gh-pages` branch). Update `base` in [vite.config.js](vite.config.js) to match your repo name (default `/recursiveMAS/`).

---

## 📄 What you do NOT need

```
✗ An API key          ✗ A server
✗ A cloud account     ✗ A subscription
✗ A GPU cluster       ✗ Any installation (for the live site)
```

---

## Further reading

- **Paper / project page:** [recursivemas.github.io](https://recursivemas.github.io)
- **Built in the style of:** [AgentHerd](https://vishalmysore.github.io/agentHerd/)

---

## License

MIT
