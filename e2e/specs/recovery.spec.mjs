import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { browser, $ } from "@wdio/globals";
import { invoke, openApp } from "../helpers/app.mjs";

const fixtures = new Set();
const fault = (id, action) => invoke("meeting_fixture", { id, action });
const meeting = (id) => invoke("get_meeting", { id });
const work = async (id) => (await invoke("meeting_work_status")).find((w) => w.meetingId === id);
const until = (check, message) => browser.waitUntil(check, { timeout: 15_000, interval: 100, timeoutMsg: message });

async function create() {
  const id = `e2e-${randomUUID()}`;
  await invoke("create_meeting", { meeting: { id, title: id, startedAt: new Date().toISOString() } });
  fixtures.add(id);
  return id;
}
async function start(id) {
  await invoke("meeting_start", { meetingId: id, modelId: `e2e-fixture:${id}`, language: "en" });
}
async function feed(id, seconds) {
  await fault(id, "feed");
  await until(async () => (await fault(id, "snapshot")).audioSamples === seconds * 16_000,
    `Expected exactly ${seconds}s of checkpointed audio`);
}
async function complete(id, count) {
  await until(async () => (await work(id))?.state === "complete", "Saved audio did not finish");
  const result = await meeting(id);
  assert.equal(result.segments.length, count, "transcript must contain no missing or duplicate rows");
  assert.deepEqual(result.segments.map((s) => [s.source, s.startMs, s.endMs, s.text]),
    Array.from({ length: count }, (_, n) => ["mic", n * 1000, (n + 1) * 1000, "Fixture speech survived."]));
  assert.equal((await work(id)).pending, 0);
}
async function retry(id) {
  // A cancellation can precede the worker thread's exit by one scheduler tick.
  await until(async () => {
    try { await invoke("meeting_retry", { id }); return true; }
    catch (e) { if (String(e).includes("previous inference call")) return false; throw e; }
  }, "Canceled worker did not release its slot");
}

describe("native recording failure recovery", () => {
  before(openApp);
  afterEach(async () => {
    for (const id of fixtures) await fault(id, "clear");
    await invoke("meeting_stop");
    for (const id of fixtures) {
      await invoke("meeting_pause", { id });
      await until(async () => (await fault(id, "snapshot")).inFlight === 0, "Fixture inference did not exit");
      await invoke("delete_meeting", { id });
      fixtures.delete(id);
    }
  });

  it("stops promptly and saves a second recording while the previous model is stuck", async () => {
    const first = await create();
    await fault(first, "stall_uncancellable");
    await start(first);
    await feed(first, 1);
    await until(async () => (await fault(first, "snapshot")).inFlight === 1, "Model did not stall");
    const before = performance.now();
    await invoke("meeting_stop");
    assert.ok(performance.now() - before < 2500, "stop waited for inference");
    assert.equal((await fault(first, "snapshot")).inFlight, 1);
    const second = await create();
    await start(second);
    await feed(second, 1);
    assert.equal((await work(second)).recording, true);
    assert.equal((await fault(second, "snapshot")).calls, 0, "a stuck call must retain the single worker slot");
    await invoke("meeting_pause", { id: first });
    await fault(first, "release");
    await invoke("meeting_stop");
    await complete(second, 1);
    assert.equal((await meeting(first)).segments.length, 0, "canceled late result must not commit");
    await retry(first);
    await complete(first, 1);
  });

  it("detects a stalled model, retains pending audio, and retries it once", async () => {
    const id = await create();
    await fault(id, "stall");
    await start(id);
    await feed(id, 1);
    await until(async () => (await fault(id, "snapshot")).inFlight === 1, "Model did not stall");
    await fault(id, "watchdog"); // shorten only this fixture's normal 90-second deadline
    await until(async () => (await work(id)).state === "error", "Supervisor missed stalled inference");
    assert.match((await work(id)).error, /stopped making progress/);
    assert.equal((await work(id)).pending, 1);
    assert.equal((await fault(id, "snapshot")).calls, 1);
    await invoke("meeting_stop");
    await fault(id, "clear");
    await retry(id);
    await complete(id, 1);
  });

  it("shows transcript storage failure in the UI and retries without duplicate rows", async () => {
    const id = await create();
    await fault(id, "fail_transcript");
    await start(id);
    await feed(id, 1);
    await until(async () => (await work(id)).state === "error", "Failed transaction was not surfaced");
    assert.equal((await meeting(id)).segments.length, 0);
    assert.equal((await work(id)).pending, 1);
    await invoke("meeting_stop");
    await $(".nav button*=meetings").click();
    const notice = $(`[data-meeting-id="${id}"]`);
    await until(async () => (await notice.getText()).includes("injected transcript storage failure"), "Storage error was not visible");
    await fault(id, "clear");
    await notice.$("button=retry saved audio").click();
    await complete(id, 1);
    await browser.reloadSession();
    await openApp();
    await complete(id, 1);
    assert.equal((await fault(id, "snapshot")).audioSamples, 16_000);
  });

  it("stops on audio write failure and reports the exact last saved position", async () => {
    const id = await create();
    await start(id);
    await feed(id, 1);
    await until(async () => (await meeting(id)).segments.length === 1, "First transcript was not saved");
    await fault(id, "fail_audio");
    await fault(id, "feed");
    await until(async () => { const s = await work(id); return s.state === "error" && !s.recording; }, "Storage failure did not release capture");
    assert.match((await work(id)).error, /injected audio storage failure/);
    assert.equal((await work(id)).micSamples, 16_000);
    assert.equal((await fault(id, "snapshot")).audioSamples, 16_000);
    await fault(id, "clear");
    await retry(id);
    await complete(id, 1);
    assert.match((await work(id)).warning, /Earlier interruption/);
  });

  it("drains the final audio callback when its device disconnects", async () => {
    const id = await create();
    await start(id);
    await feed(id, 1);
    await fault(id, "disconnect"); // atomically queue a final callback and disconnect
    await until(async () => { const s = await work(id); return s.state === "error" && !s.recording; }, "Disconnect did not release capture");
    assert.match((await work(id)).error, /injected audio disconnect/);
    assert.equal((await fault(id, "snapshot")).audioSamples, 32_000);
    await fault(id, "clear");
    await retry(id);
    await complete(id, 2);
  });

  it("recovers committed notes, transcript, and pending audio after an abrupt process kill", async () => {
    const id = await create();
    await start(id);
    await feed(id, 1);
    await until(async () => (await meeting(id)).segments.length === 1, "First transcript was not saved");
    await invoke("update_meeting_notes", { id, notes: "Saved before the crash." });
    await fault(id, "stall_uncancellable");
    await feed(id, 2);
    await until(async () => (await fault(id, "snapshot")).inFlight === 1, "Second inference did not stall");
    const { processId } = await fault(id, "snapshot");
    // This PID comes only from the isolated automation binary's guarded command.
    process.kill(processId, "SIGKILL");
    await browser.reloadSession();
    await openApp();
    await complete(id, 2);
    assert.equal((await meeting(id)).notes, "Saved before the crash.");
    assert.ok((await meeting(id)).endedAt);
    assert.equal((await fault(id, "snapshot")).audioSamples, 32_000);
    assert.match((await work(id)).warning, /Recording was interrupted/);
    await browser.reloadSession();
    await openApp();
    await complete(id, 2);
  });
});
