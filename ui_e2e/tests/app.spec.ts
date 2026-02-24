import { test, expect } from "@playwright/test";

const appUrl = process.env.STREAMLIT_APP_URL || "";
const brainSlug = process.env.BRAINS_E2E_BRAIN_SLUG || "brilliant_directories";

const statsPath = new RegExp(`/v1/brains/${brainSlug}/stats`);

const appLoadedSelector = "text=TEST_HOOK:APP_LOADED";
const statsOkSelector = "text=TEST_HOOK:STATS_OK";

const isReady = () => Boolean(appUrl);

test.skip(!isReady(), "STREAMLIT_APP_URL not set");

test("loads UI and hits stats endpoint", async ({ page }) => {
  const statsResponses: number[] = [];

  page.on("response", (resp) => {
    if (statsPath.test(resp.url())) {
      statsResponses.push(resp.status());
    }
  });

  await page.goto(appUrl, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);

  await expect(page.locator(appLoadedSelector)).toBeVisible();
  await expect(page.locator(statsOkSelector)).toBeVisible();

  await expect.poll(() => statsResponses.length > 0).toBeTruthy();
  expect(statsResponses.some((code) => code === 200)).toBeTruthy();

  await expect(page.locator("text=502")).toHaveCount(0);
  await expect(page.locator("text=Bad Gateway")).toHaveCount(0);
});

test("refresh keeps stats visible", async ({ page }) => {
  await page.goto(appUrl, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(2000);

  await expect(page.locator(appLoadedSelector)).toBeVisible();
  await expect(page.locator(statsOkSelector)).toBeVisible();

  const refreshButton = page.getByRole("button", { name: /refresh/i });
  if (await refreshButton.count()) {
    await refreshButton.first().click();
  } else {
    await page.waitForTimeout(3000);
  }

  await expect(page.locator(statsOkSelector)).toBeVisible();
  await expect(page.locator("text=502")).toHaveCount(0);
  await expect(page.locator("text=Bad Gateway")).toHaveCount(0);
});
