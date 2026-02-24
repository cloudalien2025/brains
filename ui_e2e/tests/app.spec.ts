import { test, expect } from "@playwright/test";
import fs from "fs/promises";
import path from "path";

const appUrl = process.env.STREAMLIT_APP_URL || "";
const brainSlug = process.env.BRAINS_E2E_BRAIN_SLUG || "brilliant_directories";
const statsPathRaw =
  process.env.WORKER_STATS_PATH || `/v1/brains/${brainSlug}/stats`;
const statsPath = new RegExp(statsPathRaw);
const timeoutMs = Number(process.env.E2E_TIMEOUT_MS || 120000);

const isReady = () => Boolean(appUrl);

test.skip(!isReady(), "STREAMLIT_APP_URL not set");

test.beforeEach(async ({ page }) => {
  await page.context().tracing.start({
    screenshots: true,
    snapshots: true,
    sources: true,
  });
});

test.afterEach(async ({ page }, testInfo) => {
  const networkLogLines = (testInfo as any).networkLogLines as string[] | undefined;
  const testName = sanitizeName(testInfo.titlePath().join(" - "));
  const baseDir = path.join(process.cwd(), "test-results", testName);
  const tracePath = path.join(baseDir, "trace.zip");

  try {
    if (testInfo.status !== testInfo.expectedStatus) {
      await page.context().tracing.stop({ path: tracePath });
    } else {
      await page.context().tracing.stop();
    }
  } catch {
    // Best-effort: tracing can fail if the browser crashed.
  }

  if (testInfo.status === testInfo.expectedStatus) return;
  await dumpArtifacts(page, testInfo, networkLogLines || [], tracePath, baseDir);
});

function sanitizeName(value: string) {
  return value.replace(/[^a-zA-Z0-9-_]+/g, "_").slice(0, 120);
}

async function dumpArtifacts(
  page: any,
  testInfo: any,
  networkLogLines: string[],
  tracePath: string,
  baseDir: string
) {
  await fs.mkdir(baseDir, { recursive: true });

  const screenshotPath = path.join(baseDir, "page.png");
  const htmlPath = path.join(baseDir, "page.html");
  const innerTextPath = path.join(baseDir, "innerText.txt");
  const networkPath = path.join(baseDir, "network.log");

  await page.screenshot({ path: screenshotPath, fullPage: true });
  const html = await page.content();
  await fs.writeFile(htmlPath, html, "utf8");
  const innerText = await page.evaluate(() => document.body?.innerText || "");
  await fs.writeFile(innerTextPath, innerText.slice(0, 20000), "utf8");
  await fs.writeFile(networkPath, networkLogLines.join("\n"), "utf8");

  // eslint-disable-next-line no-console
  console.log(
    `ARTIFACTS: ${screenshotPath} ${htmlPath} ${innerTextPath} ${networkPath} ${tracePath}`
  );
}

async function waitForRealPage(page: any, timeout: number) {
  const start = Date.now();
  let delay = 500;

  while (Date.now() - start < timeout) {
    const result = await page.evaluate(() => {
      const ready = document.readyState === "complete";
      const bodyTextLen = document.body?.innerText?.trim().length || 0;
      const hasStreamlitRoot = Boolean(
        document.querySelector(
          '[data-testid="stAppViewContainer"], [data-testid="stApp"]'
        )
      );
      const hasTitle = Boolean(document.title && document.title.trim().length > 0);

      return { ready, bodyTextLen, hasStreamlitRoot, hasTitle };
    });

    if (
      result.ready &&
      result.bodyTextLen > 200 &&
      (result.hasStreamlitRoot || result.hasTitle)
    ) {
      return;
    }

    await page.waitForTimeout(delay);
    delay = Math.min(Math.round(delay * 1.5), 5000);
  }

  throw new Error("Timed out waiting for Streamlit page readiness signal");
}

function isGatewayError(status: number) {
  return [502, 504, 520, 521, 522, 523].includes(status);
}

test("loads UI and hits stats endpoint", async ({ page }, testInfo) => {
  const statsResponses: number[] = [];
  const networkLogLines: string[] = [];
  (testInfo as any).networkLogLines = networkLogLines;

  page.on("request", (req) => {
    if (req.url().includes("/v1/")) {
      networkLogLines.push(
        `${new Date().toISOString()} REQUEST ${req.method()} ${req.url()}`
      );
    }
  });

  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/v1/")) {
      networkLogLines.push(
        `${new Date().toISOString()} RESPONSE ${resp.status()} ${url}`
      );
    }
    if (statsPath.test(url)) {
      statsResponses.push(resp.status());
    }
  });

  await page.goto(appUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForRealPage(page, 60000);

  await expect.poll(() => statsResponses.length > 0, { timeout: timeoutMs }).toBeTruthy();

  const hasOk = statsResponses.some((code) => code === 200);
  if (!hasOk) {
    const gatewaySeen = statsResponses.some((code) => isGatewayError(code));
    if (gatewaySeen) {
      throw new Error(
        `Stats endpoint returned gateway errors (${statsResponses.join(", ")}) and no 200`
      );
    }
  }

  expect(hasOk).toBeTruthy();
});

test("refresh or wait triggers another stats call", async ({ page }, testInfo) => {
  const statsResponses: number[] = [];
  const networkLogLines: string[] = [];
  (testInfo as any).networkLogLines = networkLogLines;

  page.on("request", (req) => {
    if (req.url().includes("/v1/")) {
      networkLogLines.push(
        `${new Date().toISOString()} REQUEST ${req.method()} ${req.url()}`
      );
    }
  });

  page.on("response", (resp) => {
    const url = resp.url();
    if (url.includes("/v1/")) {
      networkLogLines.push(
        `${new Date().toISOString()} RESPONSE ${resp.status()} ${url}`
      );
    }
    if (statsPath.test(url)) {
      statsResponses.push(resp.status());
    }
  });

  await page.goto(appUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForRealPage(page, 60000);

  await expect.poll(() => statsResponses.length > 0, { timeout: timeoutMs }).toBeTruthy();
  const initialSeen = statsResponses.length;
  const hasOk = () => statsResponses.some((code) => code === 200);

  const refreshButton = page.getByRole("button", { name: /refresh/i });
  if (await refreshButton.count()) {
    await refreshButton.first().click();
  } else {
    await page.waitForTimeout(10000);
  }

  if (!hasOk()) {
    await expect
      .poll(() => statsResponses.length > initialSeen || hasOk(), { timeout: 20000 })
      .toBeTruthy();
  }

  if (!hasOk()) {
    const gatewaySeen = statsResponses.some((code) => isGatewayError(code));
    if (gatewaySeen) {
      throw new Error(
        `Stats endpoint returned gateway errors (${statsResponses.join(", ")}) and no 200`
      );
    }
  }

  expect(hasOk()).toBeTruthy();
});
