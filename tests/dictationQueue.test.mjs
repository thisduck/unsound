import { test } from "node:test";
import assert from "node:assert/strict";
import { createDictationQueue } from "../src/dictationQueue.ts";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const settle = () => new Promise(setImmediate);

function harness(overrides = {}) {
  const calls = [], errors = [], states = [];
  const queue = createDictationQueue({
    start: async (take) => { calls.push(`start:${take.id}`); },
    finish: async (take, cancelled) => {
      calls.push(`${cancelled ? "cancel" : "finish"}:${take.id}`);
      return cancelled ? null : `audio:${take.id}`;
    },
    process: async (take, audio) => { calls.push(`process:${take.id}:${audio}`); },
    changed: (state) => { states.push(state); },
    failed: (take, error) => { errors.push([take.id, String(error)]); },
    ...overrides,
  });
  return { queue, calls, errors, states };
}

test("capture continues during transcription, cleanup and delivery, preserving spoken order", async () => {
  const transcribe = deferred(), cleanup = deferred(), delivery = deferred();
  const processed = [], typed = [];
  const { queue } = harness({ process: async (take, audio) => {
    processed.push([take.id, audio, take.style]);
    if (take.id === "first") {
      await transcribe.promise;
      await cleanup.promise;
      await delivery.promise;
    }
    typed.push(take.id);
  } });
  queue.start({ id: "first", style: "casual" });
  queue.stop();
  await settle();
  queue.start({ id: "second", style: "formal" });
  await settle();
  assert.equal(queue.state().active.id, "second");
  queue.stop();
  await settle();
  assert.equal(queue.state().pending, 2);
  assert.deepEqual(processed, [["first", "audio:first", "casual"]]);
  transcribe.resolve();
  await settle();
  queue.start({ id: "third", style: "casual" });
  queue.stop();
  await settle();
  cleanup.resolve();
  await settle();
  assert.equal(queue.state().pending, 3);
  assert.deepEqual(typed, []);
  delivery.resolve();
  await settle();
  assert.deepEqual(typed, ["first", "second", "third"]);
  assert.deepEqual(processed, [
    ["first", "audio:first", "casual"], ["second", "audio:second", "formal"], ["third", "audio:third", "casual"],
  ]);
  assert.deepEqual(queue.state(), { active: null, capturing: false, pending: 0 });
});

test("an older completion leaves the new take recording", async () => {
  const processing = deferred();
  const { queue } = harness({ process: async () => processing.promise });
  queue.start({ id: "first" });
  queue.stop();
  await settle();
  queue.start({ id: "second" });
  await settle();
  processing.resolve();
  await settle();
  assert.deepEqual(queue.state(), { active: { id: "second" }, capturing: true, pending: 0 });
  queue.stop();
  await settle();
  assert.equal(queue.state().pending, 0);
});

test("rapid down/up/down/up during mic startup retains both takes exactly once", async () => {
  const starting = deferred();
  const starts = [];
  const { queue, calls } = harness({ start: async (take) => {
    starts.push(take.id);
    if (take.id === "first") await starting.promise;
  } });
  queue.start({ id: "first" });
  queue.start({ id: "repeat keydown" });
  queue.stop();
  queue.stop();
  queue.start({ id: "second" });
  queue.stop();
  await settle();
  assert.deepEqual(starts, ["first"]);
  assert.deepEqual(calls, []);
  starting.resolve();
  await settle();
  assert.deepEqual(starts, ["first", "second"]);
  assert.deepEqual(calls.filter(c => c.startsWith("finish:")), ["finish:first", "finish:second"]);
  assert.deepEqual(calls.filter(c => c.startsWith("process:")), ["process:first:audio:first", "process:second:audio:second"]);
  assert.deepEqual(queue.state(), { active: null, capturing: false, pending: 0 });
});

test("the next capture waits for audio handoff, but never for processing", async () => {
  const stopping = deferred(), processing = deferred();
  const { queue, calls } = harness({
    finish: async (take) => {
      if (take.id === "first") await stopping.promise;
      return take.id;
    },
    process: async () => processing.promise,
  });
  queue.start({ id: "first" });
  queue.stop();
  queue.start({ id: "second" });
  await settle();
  assert.deepEqual(calls, ["start:first"]);
  stopping.resolve();
  await settle();
  assert.deepEqual(calls, ["start:first", "start:second"]);
  assert.equal(queue.state().active.id, "second");
  processing.resolve();
  queue.stop();
  await settle();
});

test("cancelling during startup discards only that take and allows an immediate restart", async () => {
  const starting = deferred();
  const { queue, calls } = harness({ start: async () => starting.promise });
  queue.start({ id: "cancelled" });
  queue.stop(true);
  queue.start({ id: "next" });
  queue.stop();
  starting.resolve();
  await settle();
  assert.deepEqual(calls, ["cancel:cancelled", "finish:next", "process:next:audio:next"]);
  assert.equal(queue.state().pending, 0);
});

test("a processing error does not drop or block subsequent takes", async () => {
  const failed = deferred(), delivered = [];
  const { queue, errors } = harness({ process: async (take) => {
    if (take.id === "failed") await failed.promise;
    delivered.push(take.id);
  } });
  queue.start({ id: "failed" });
  queue.stop();
  queue.start({ id: "next" });
  queue.stop();
  await settle();
  failed.reject(new Error("transcription unavailable"));
  await settle();
  assert.deepEqual(delivered, ["next"]);
  assert.deepEqual(errors, [["failed", "Error: transcription unavailable"]]);
  assert.deepEqual(queue.state(), { active: null, capturing: false, pending: 0 });
});

test("failed starts, failed stops and short takes release their queue slots", async () => {
  const finished = [], processed = [];
  const { queue, errors } = harness({
    start: async (take) => { if (take.id === "start-fails") throw new Error("mic unavailable"); },
    finish: async (take) => {
      finished.push(take.id);
      if (take.id === "stop-fails") throw new Error("capture failed");
      return take.id === "short" ? null : take.id;
    },
    process: async (take) => { processed.push(take.id); },
  });
  for (const id of ["start-fails", "stop-fails", "short", "next"]) {
    queue.start({ id });
    queue.stop();
  }
  await settle();
  assert.deepEqual(finished, ["stop-fails", "short", "next"]);
  assert.deepEqual(processed, ["next"]);
  assert.deepEqual(errors.map(([id]) => id), ["start-fails", "stop-fails"]);
  assert.deepEqual(queue.state(), { active: null, capturing: false, pending: 0 });
});
