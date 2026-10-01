import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// base "./": every asset URL is relative, so the app works behind a path-prefixed reverse proxy (upstream f0e1b9e).
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  build: { outDir: "dist", emptyOutDir: true, assetsDir: "assets", target: "es2022" },
  server: { proxy: { "/metrics": "http://127.0.0.1:8091", "/v1": "http://127.0.0.1:8091" } },
})
