import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Flask API runs on :5000; Vite proxies /api to it so the
// browser sees a single origin (no CORS, auth header works for <img> fetches).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_API_TARGET || "http://localhost:5000", changeOrigin: true } },
  },
});
