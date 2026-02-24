import { defineConfig } from "@playwright/test";

const streamlitUrl = process.env.STREAMLIT_APP_URL || "";

export default defineConfig({
  testDir: "./tests",
  timeout: 60000,
  expect: { timeout: 15000 },
  use: {
    baseURL: streamlitUrl,
    headless: true,
    viewport: { width: 1400, height: 900 },
  },
  reporter: [["list"]],
});
