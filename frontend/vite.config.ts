import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const backend = loadEnv(mode, process.cwd(), "").TOFU_BACKEND_URL || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": backend,
        "/outputs": backend,
        "/uploads": backend,
      },
    },
  };
});
