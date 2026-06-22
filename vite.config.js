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
