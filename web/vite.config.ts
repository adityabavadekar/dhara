import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(import.meta.dirname, './src') } },
  // MapLibre 6 loads its ES-module worker relative to its own file; pre-bundling breaks that.
  optimizeDeps: { exclude: ['maplibre-gl'] },
  server: {
    port: 5173,
    proxy: { '/api': { target: process.env.DHARA_API ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
