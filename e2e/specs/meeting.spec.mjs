import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { browser, $ } from "@wdio/globals";

import { invoke, openApp } from "../helpers/app.mjs";

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
    await $(".nav button*=meetings").click();
    await $(".meet-row-title=E2E interrupted meeting").click();
    const notes = $("textarea.meet-notes");
    await notes.setValue("Decision: edited in the native notes UI.");
    await browser.waitUntil(async () => (await invoke("get_meeting", { id })).notes === "Decision: edited in the native notes UI.", {
      timeoutMsg: "Notes typed in the native UI did not reach SQLite",
    });
    // Delete/new WebDriver session restarts the native application. Startup
    // recovery closes unfinished metadata, unlike a mere webview refresh.
    await browser.reloadSession();
    await openApp();
    meeting = await invoke("get_meeting", { id });
    assert.equal(meeting.notes, "Decision: edited in the native notes UI.");
    assert.ok(meeting.endedAt, "native startup should recover the open meeting");
    assert.deepEqual(meeting.segments, []);
  });
});
