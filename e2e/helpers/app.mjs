import { browser, $ } from "@wdio/globals";

// Shared native webview helpers. Application commands use real Tauri IPC.
export async function invoke(command, args) {
  const result = await browser.executeAsync((command, args, done) => {
    window.__TAURI_INTERNALS__.invoke(command, args)
      .then((value) => done({ value }), (error) => done({ error: String(error) }));
  }, command, args);
  if (result.error) throw new Error(result.error);
  return result.value;
}

export async function openApp() {
  await browser.waitUntil(() => browser.execute(() => Boolean(window.__TAURI_INTERNALS__)), {
    timeoutMsg: "Tauri IPC did not initialize",
  });
  // Onboarding can appear on a fresh isolated profile. Exercise its real skip action.
  const skip = $("button*=skip — set up later");
  await browser.waitUntil(async () => (await skip.isExisting()) || (await $(".head .nav").isExisting()));
  if (await skip.isExisting()) await skip.click();
  await $(".head .nav").waitForDisplayed();
}

