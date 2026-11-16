import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // or your preferred port
    open: true, // auto-open browser
    hmr: {
      overlay: true // show errors in browser overlay
    },
    watch: {
      usePolling: true, // useful for some systems
      interval: 100 // check for changes every 100ms
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  // Optional: clear cache on build
  build: {
    sourcemap: true,
  }
})