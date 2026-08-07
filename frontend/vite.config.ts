/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const backend = loadEnv(mode, process.cwd(), "").TOFU_BACKEND_URL || "http://localhost:8000";
  return {
    plugins: [react()],
    build: {
      sourcemap: true,
      rollupOptions: {
        output: {
          // Keep the project gate/editor shell small enough to display before
          // optional design tooling.  Component-level lazy imports handle
          // infrequent panels; these stable vendor chunks avoid re-downloading
          // React/MUI/icon/WebGL code with the application shell.
          manualChunks(id) {
            if (!id.includes("node_modules")) return undefined;
            if (id.includes("@mui") || id.includes("@emotion")) return "vendor-mui";
            if (id.includes("react-icons") || id.includes("lucide-react")) return "vendor-icons";
            if (id.includes("/ogl/")) return "vendor-webgl";
            if (id.includes("/react/") || id.includes("react-dom")) return "vendor-react";
            return "vendor";
          },
        },
      },
    },
    server: {
      port: 5173,
      proxy: {
        "/api": backend,
        "/outputs": backend,
        "/uploads": backend,
      },
    },
    test: {
      environment: "happy-dom",
      setupFiles: ["./src/test-setup.ts"],
    },
  };
});
