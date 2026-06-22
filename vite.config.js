import { defineConfig } from 'vite';

export default defineConfig({
  // Change this to match your GitHub Pages repo name (e.g. '/recursiveMAS/').
  // Use '/' if deploying to a root user/org site.
  base: '/recursiveMASDemo/',
  optimizeDeps: {
    exclude: ['@mlc-ai/web-llm'],
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    target: 'esnext',
    sourcemap: false,
    // Multi-page: the single-browser demo (index) + the WebRTC P2P latent demo (p2p).
    rollupOptions: {
      input: { main: 'index.html', p2p: 'p2p.html' },
    },
  },
  server: {
    port: 3001,
    open: true,
    headers: {
      // Required so WebLLM/WebGPU can use SharedArrayBuffer.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
