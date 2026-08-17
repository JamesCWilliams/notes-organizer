import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// Absolute path to the repo's saves/ folder, where notes live. Resolved here
// rather than in the app because Tauri exposes app-data/home/resource dirs but
// no working directory, so the frontend has no way to locate the repo at
// runtime. npm scripts run from the repo root, so cwd is that root.
// @ts-expect-error process is a nodejs global
const savesDir = `${process.cwd()}/saves`;

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  define: { __SAVES_DIR__: JSON.stringify(savesDir) },

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
