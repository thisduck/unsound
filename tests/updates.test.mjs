import { test } from "node:test";
import assert from "node:assert/strict";
import { createUpdateController } from "../src/updateController.ts";
import { markUnsaved, trackWork, updateWorkBlocker } from "../src/updateActivity.ts";
const update = { version: "0.9.2", currentVersion: "0.9.1", notes: "New features" };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
function harness(overrides = {}) {
  let installs = 0;
  const c = createUpdateController({ check: async () => update, download: async () => {}, install: async () => { installs++; }, blocker: () => null, changed: () => {}, ...overrides });
  return { c, installs: () => installs };
}

test("update checks coalesce and report offline failures without claiming up to date", async () => {
  const pending = deferred(); let calls = 0;
  const { c } = harness({ check: () => { calls++; return pending.promise; } });
  const first = c.check(); await c.check(); assert.equal(calls, 1);
  pending.reject(new Error("offline")); await first;
  assert.equal(c.state.phase, "idle"); assert.match(c.state.error, /offline/);
});

test("up-to-date is shown only after a successful check with no newer version", async () => {
  const { c } = harness({ check: async () => null });
  await c.check(); assert.equal(c.state.phase, "current"); assert.equal(c.state.error, null);
});

test("failed verification cannot install and downloads can be retried", async () => {
  let tries = 0;
  const { c, installs } = harness({ download: async () => { if (++tries === 1) throw new Error("invalid signature"); } });
  await c.check(); await c.download(); await c.install();
  assert.equal(installs(), 0); assert.equal(c.state.phase, "available"); assert.match(c.state.error, /signature/);
  await c.download(); assert.equal(c.state.phase, "ready"); assert.equal(installs(), 0);
  await c.install(); assert.equal(installs(), 1);
});

test("recording that starts during download blocks restart until work finishes", async () => {
  const pending = deferred(); let busy = false;
  const { c, installs } = harness({ download: () => pending.promise, blocker: () => busy ? "Recording" : null });
  await c.check(); const download = c.download(); busy = true;
  c.progress(40, 100); assert.equal(c.state.downloaded, 40);
  pending.resolve(); await download;
  await c.install(); assert.equal(installs(), 0); assert.equal(c.state.phase, "ready");
  busy = false; await c.install(); assert.equal(installs(), 1);
});

test("checking cannot discard a verified download and duplicate install clicks coalesce", async () => {
  const pending = deferred(); let calls = 0;
  const { c } = harness({ install: () => { calls++; return pending.promise; } });
  await c.check(); await c.download(); await c.check(); assert.equal(c.state.phase, "ready");
  const first = c.install(); await c.install(); assert.equal(calls, 1);
  pending.reject(new Error("disk full")); await first;
  assert.equal(c.state.phase, "ready"); assert.match(c.state.error, /disk full/);
});

test("unsaved notes and failed pending work hold and release the restart blocker", async () => {
  markUnsaved("update-test", true); assert.match(updateWorkBlocker(), /Save/);
  markUnsaved("update-test", false);
  const pending = deferred(); const work = trackWork(() => pending.promise);
  assert.match(updateWorkBlocker(), /Wait/);
  pending.reject(new Error("storage unavailable")); await assert.rejects(work);
  assert.equal(updateWorkBlocker(), null);
});
