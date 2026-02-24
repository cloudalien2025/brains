import { defineConfig } from "@playwright/test";

const streamlitUrl = process.env.STREAMLIT_APP_URL || "";
const timeoutMs = Number(process.env.E2E_TIMEOUT_MS || 120000);

export default defineConfig({
  testDir: "./tests",
  outputDir: "/tmp/playwright-results",
  timeout: timeoutMs,
  expect: { timeout: 15000 },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: streamlitUrl,
    headless: true,
    viewport: { width: 1400, height: 900 },
    navigationTimeout: 60000,
    chromiumSandbox: false,
    launchOptions: {
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  reporter: [["list"]],
});
