/// <reference types="vitest/config" />
// The pipeline service's browser app (design/web-app.md §4). The service serves the build at /ui (BEQ_SERVICE_UI);
// `npm run dev` proxies the API to a service running locally (BEQ_SERVICE_URL, default http://127.0.0.1:8080).
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const service = process.env.BEQ_SERVICE_URL ?? 'http://127.0.0.1:8080'

export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  server: {
    proxy: { '/v1': { target: service, changeOrigin: true } },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
})
