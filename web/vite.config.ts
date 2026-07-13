import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend =
  process.env.BACKEND_URL ?? "http://127.0.0.1:8222";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: Object.fromEntries(
      [
        "/transcribe",
        "/instruments",
        "/config",
        "/auralize",
        "/health",
        "/soundfonts",
      ].map((path) => [
        path,
        { target: backend, changeOrigin: true, secure: true },
      ]),
    ),
  },
  build: {
    // Build straight into the Python package so the frontend ships inside
    // the wheel (see [tool.hatch.build] artifacts in pyproject.toml).
    outDir: "../muscriptor/web_dist",
    emptyOutDir: true,
  },
});
