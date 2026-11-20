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
  const skip = $("button=Skip for now");
  await browser.waitUntil(async () => (await skip.isExisting()) || (await $(".topbar-nav").isExisting()));
  if (await skip.isExisting()) await skip.click();
  await $(".topbar-nav").waitForDisplayed();
  // The launch-time permissions check appears for set-up profiles missing
  // Microphone or Accessibility; mark it seen so restarts aren't blocked.
  await browser.execute(() =>
    localStorage.setItem("unsound.permissions.reviewed", JSON.stringify(["microphone", "accessibility"])),
  );
  const later = $("button=Not now");
  if (await later.isExisting()) await later.click();
}

