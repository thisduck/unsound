import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { browser, $ } from "@wdio/globals";
import { invoke, openApp } from "../helpers/app.mjs";

describe("meeting follow-through", () => {
  const id = `e2e-${randomUUID()}`;
  const title = `Follow-up ${id}`;
  let item;
  before(async () => {
    await openApp();
    await invoke("create_meeting", { meeting: { id, title, startedAt: "2026-10-15T10:00:00Z" } });
    item = await invoke("save_action_item", { item: { meetingId: id, text: "Send the test draft", remindAt: Date.now() - 1000 } });
    await browser.refresh(); await openApp();
  });
  after(async () => { await invoke("delete_meeting", { id }); });
  it("shows a due reminder, persists ignore, and allows reopening and completion", async () => {
    await $(".reminder-inbox").waitForDisplayed();
    await $(".reminder-inbox button=ignore").click();
    await browser.waitUntil(async () => (await invoke("list_action_items")).find(i => i.id === item.id)?.ignored === true);
    await browser.reloadSession(); await openApp();
    const ignored = (await invoke("list_action_items")).find(i => i.id === item.id);
    assert.equal(ignored.ignored, true); assert.equal(ignored.remindAt, null);
    await $(".nav button*=meetings").click();
    await $(`.meet-row-title=${title}`).click();
    await $(".follow-up button=reopen").click();
    await $(".follow-up button=mark done").waitForDisplayed();
    await $(".follow-up button=mark done").click();
    await browser.waitUntil(async () => (await invoke("list_action_items")).find(i => i.id === item.id)?.done === true);
    assert.equal((await invoke("due_action_reminders")).some(i => i.id === item.id), false);
  });
});
