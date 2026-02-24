import { chromium } from "@playwright/test";
import fs from "fs/promises";
import path from "path";

const appUrl = process.env.STREAMLIT_APP_URL || "";
const brainSlug = process.env.BRAINS_E2E_BRAIN_SLUG || "brilliant_directories";
const statsPathRaw =
  process.env.WORKER_STATS_PATH || `/v1/brains/${brainSlug}/stats`;
const statsPath = new RegExp(statsPathRaw);
const timeoutMs = Number(process.env.E2E_TIMEOUT_MS || 120000);

function sanitizeName(value: string) {
  return value.replace(/[^a-zA-Z0-9-_]+/g, "_").slice(0, 120);
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

async function dumpArtifacts(
  page: any,
  networkLogLines: string[],
  targetDir: string
) {
  await fs.mkdir(targetDir, { recursive: true });

  const screenshotPath = path.join(targetDir, "page.png");
  const htmlPath = path.join(targetDir, "page.html");
  const innerTextPath = path.join(targetDir, "innerText.txt");
  const networkPath = path.join(targetDir, "network.log");

  await page.screenshot({ path: screenshotPath, fullPage: true });
  const html = await page.content();
  await fs.writeFile(htmlPath, html, "utf8");
  const innerText = await page.evaluate(() => document.body?.innerText || "");
  await fs.writeFile(innerTextPath, innerText.slice(0, 20000), "utf8");
  await fs.writeFile(networkPath, networkLogLines.join("\n"), "utf8");

  // eslint-disable-next-line no-console
  console.log(
    `ARTIFACTS: ${screenshotPath} ${htmlPath} ${innerTextPath} ${networkPath}`
  );
}

async function main() {
  if (!appUrl) {
    throw new Error("STREAMLIT_APP_URL is required");
  }

  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  const page = await browser.newPage();

  const networkLogLines: string[] = [];
  const workerCalls: { status: number; url: string }[] = [];
  const consoleErrors: string[] = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });

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
      workerCalls.push({ status: resp.status(), url });
    }
  });

  await page.goto(appUrl, { waitUntil: "domcontentloaded", timeout: 60000 });
  await waitForRealPage(page, 60000);
  await page.waitForTimeout(Math.min(10000, Math.max(0, timeoutMs - 10000)));

  const title = await page.title();
  const innerText = await page.evaluate(() => document.body?.innerText || "");

  // eslint-disable-next-line no-console
  console.log(`TITLE: ${title}`);
  // eslint-disable-next-line no-console
  console.log(`INNER_TEXT: ${innerText.slice(0, 2000)}`);

  if (consoleErrors.length) {
    // eslint-disable-next-line no-console
    console.log("CONSOLE_ERRORS:");
    for (const line of consoleErrors) {
      // eslint-disable-next-line no-console
      console.log(line);
    }
  }

  if (workerCalls.length) {
    // eslint-disable-next-line no-console
    console.log("WORKER_CALLS:");
    for (const call of workerCalls) {
      // eslint-disable-next-line no-console
      console.log(`${call.status} ${call.url}`);
    }
  } else {
    // eslint-disable-next-line no-console
    console.log("WORKER_CALLS: none");
  }

  const targetDir = path.join(
    process.cwd(),
    "test-results",
    sanitizeName("diagnose"),
    new Date().toISOString().replace(/[:.]/g, "-")
  );
  await dumpArtifacts(page, networkLogLines, targetDir);

  await browser.close();
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exitCode = 1;
});
