import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { browser, $ } from "@wdio/globals";

// Use the real Tauri IPC surface: no frontend mocks or test-only data commands.
async function invoke(command, args) {
  const result = await browser.executeAsync((command, args, done) => {
    window.__TAURI_INTERNALS__.invoke(command, args)
      .then((value) => done({ value }), (error) => done({ error: String(error) }));
  }, command, args);
  if (result.error) throw new Error(result.error);
  return result.value;
}

async function openApp() {
  await browser.waitUntil(() => browser.execute(() => Boolean(window.__TAURI_INTERNALS__)), {
    timeoutMsg: "Tauri IPC did not initialize",
  });
  // Onboarding can appear on a fresh isolated profile. Exercise its real skip action.
  const skip = $("button*=skip — set up later");
  await browser.waitUntil(async () => (await skip.isExisting()) || (await $(".head .nav").isExisting()));
  if (await skip.isExisting()) await skip.click();
  await $(".head .nav").waitForDisplayed();
}

describe("native meeting persistence", () => {
  const id = `e2e-${randomUUID()}`;
  before(openApp);
  after(async () => { await invoke("delete_meeting", { id }); });

  it("navigates the actual webview", async () => {
    await $(".nav button*=meetings").click();
    assert.match(await $(".nav button.on").getText(), /meetings/);
    await $(".nav button*=tasks").click();
    assert.match(await $(".nav button.on").getText(), /tasks/);
    await $(".nav button*=dictate").click();
    assert.equal(await $(".nav button.on").getText(), "dictate");
  });

  it("retains saved notes across webview reload and native app restart", async () => {
    await invoke("create_meeting", { meeting: {
      id, title: "E2E interrupted meeting", startedAt: "2026-01-01T00:00:00Z",
    } });
    await invoke("update_meeting_notes", { id, notes: "Decision: ship the recovery tests." });
    await browser.refresh();
    await openApp();
    let meeting = await invoke("get_meeting", { id });
    assert.equal(meeting.notes, "Decision: ship the recovery tests.");
    assert.equal(meeting.endedAt, null, "a webview reload does not finalize recording metadata");
    // Delete/new WebDriver session restarts the native application. Startup
    // recovery closes unfinished metadata, unlike a mere webview refresh.
    await browser.reloadSession();
    await openApp();
    meeting = await invoke("get_meeting", { id });
    assert.equal(meeting.notes, "Decision: ship the recovery tests.");
    assert.ok(meeting.endedAt, "native startup should recover the open meeting");
    assert.deepEqual(meeting.segments, []);
  });
});
