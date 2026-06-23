import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages project site: https://<user>.github.io/nba-win-baseline/
const base = process.env.VITE_BASE_PATH ?? "/nba-win-baseline/";

export default defineConfig({
  plugins: [react()],
  base,
});
