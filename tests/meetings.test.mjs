import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import ts from "typescript";
import React from "react";
import { act, create } from "react-test-renderer";

// Mount the real Meetings controller; replace native IPC and leaf views only.
// Node strips .ts types but needs a JSX transform for this component.
const componentUrl = new URL("../src/Meetings.tsx", import.meta.url).href;
const virtual = (source) => ({ url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true });
const hooks = registerHooks({
  resolve(specifier, context, next) {
    if (context.parentURL === componentUrl) {
      if (specifier === "./api") return virtual("export const { api, on } = globalThis.meetingsTest;");
      if (specifier === "@tauri-apps/api/event") return virtual("export const listen = globalThis.meetingsTest.listen;");
      const name = specifier.split("/").at(-1);
      if (["MeetingSetup", "DetailView", "ListView", "LiveView"].includes(name)) {
        // JSX runtime resolves from a file URL, rather than a data URL.
        return virtual(`import { createElement } from ${JSON.stringify(import.meta.resolve("react"))};
          export function ${name}(props) { return createElement(${JSON.stringify(name)}, props); }`);
      }
      if (name === "TaskCard") return virtual("export const mergeTask = (tasks, task) => [...tasks, task];");
    }
    return next(specifier, context);
  },
  load(url, context, next) {
    if (url === componentUrl) return {
      format: "module", shortCircuit: true,
      source: ts.transpileModule(readFileSync(new URL(url), "utf8"), {
        compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
      }).outputText,
    };
    return next(url, context);
  },
});

let fixture;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.localStorage = { getItem: () => null, setItem() {} };
globalThis.window = {
  setInterval: (fn) => { fixture.poll = fn; return 1; },
  clearInterval() {},
};
globalThis.meetingsTest = {
  api: new Proxy({}, { get: (_, name) => (...args) => fixture.api[name](...args) }),
  on: new Proxy({}, { get: (_, name) => (cb) => {
    fixture.events[name] = cb;
    return Promise.resolve(() => {});
  } }),
  listen: () => Promise.resolve(() => {}),
};
const { Meetings } = await import(componentUrl);
hooks.deregister();

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};
const models = [
  { id: "speech", name: "Speech", kind: "stt", downloaded: true },
  { id: "notes", name: "Notes", kind: "llm", downloaded: true },
  { id: "voice", name: "Voice", kind: "diarize", downloaded: true },
  { id: "vad-silero", kind: "vad", downloaded: true },
];

async function mount(t) {
  const rows = new Map(), meetings = new Map(), jobs = new Map(), starts = [];
  fixture = {
    rows, meetings, jobs, starts, events: {}, active: null,
    api: {
      listMeetings: async () => [...meetings.values()],
      getMeeting: async (id) => meetings.get(id),
      getSettings: async () => ({ assistant: {}, commandKeys: [], meetings: { summaryModel: "notes" } }),
      systemAudioSupported: async () => true,
      hideMeetingPrompt: async () => {},
      meetingWorkStatus: async () => [...rows.values()],
      createMeeting: async (m) => meetings.set(m.id, m),
      meetingStart: async (id) => {
        assert.equal(fixture.active, null, "only one capture can own the recorder");
        fixture.active = id;
        starts.push(id);
        rows.set(id, { meetingId: id, state: "recording", recording: true, pending: 0 });
        return meetings.get(id);
      },
      meetingStop: async (id) => {
        assert.equal(fixture.active, id);
        fixture.active = null;
        rows.set(id, { meetingId: id, state: "finishing", recording: false, pending: 1 });
        fixture.events.meetingStopped(id);
        return meetings.get(id);
      },
      listTasks: async () => [],
      summarizeMeeting: (id, model) => {
        assert.equal(jobs.has(id), false, "one summary per meeting");
        const gate = deferred();
        jobs.set(id, { ...gate, model });
        return gate.promise;
      },
    },
  };
  let renderer;
  await act(async () => { renderer = create(React.createElement(Meetings, { models, language: "en", onModelsChanged() {} })); });
  t.after(async () => { await act(async () => renderer.unmount()); });
  return {
    ...fixture,
    pill: () => renderer.root.findAllByType("button").find(b => b.props.className.startsWith("pill")),
    view: (name) => renderer.root.findByType(name),
    click: async () => { await act(async () => {
      const pill = renderer.root.findAllByType("button").find(b => b.props.className.startsWith("pill"));
      assert.equal(pill.props.disabled, false, "recording control must remain enabled");
      await pill.props.onClick();
    }); },
    emit: async (event, payload) => { await act(async () => fixture.events[event](payload)); },
    completeTranscript: async (id) => { await act(async () => {
      rows.set(id, { meetingId: id, state: "complete", recording: false, pending: 0 });
      await fixture.poll();
    }); },
    finishSummary: async (id, error) => { await act(async () => {
      if (error) jobs.get(id).reject(error);
      else jobs.get(id).resolve({ ...meetings.get(id), summary: `Notes for ${id}` });
    }); },
  };
}

for (const start of ["button", "meetingToggle", "meetingStartRequest"]) {
  test(`a new meeting starts via ${start} while previous notes generate`, async (t) => {
    const h = await mount(t);
    await h.click();
    const first = h.starts[0];
    await h.click();
    await h.completeTranscript(first);
    assert.ok(h.jobs.has(first));
    assert.equal(h.view("DetailView").props.summarizing, true);
    assert.equal(h.pill().props.disabled, false);
    if (start === "button") await h.click();
    else await h.emit(start);
    const second = h.starts[1];
    assert.ok(second);
    await h.finishSummary(first);
    assert.equal(h.view("LiveView").props.meetingId, second);
    assert.match(h.pill().props.className, /live/);
    await h.click();
    assert.equal(fixture.active, null, "the second meeting still has a working stop control");
  });
}

test("a prior transcript can finish and generate notes during a new recording", async (t) => {
  const h = await mount(t);
  await h.click();
  const first = h.starts[0];
  await h.click();
  await h.click();
  const second = h.starts[1];
  await h.completeTranscript(first);
  assert.ok(h.jobs.has(first), "automatic notes must not depend on the selected meeting");
  assert.equal(h.view("LiveView").props.meetingId, second);
  await h.finishSummary(first, new Error("generation failed"));
  assert.equal(h.view("LiveView").props.meetingId, second);
  await h.click();
  await h.completeTranscript(second);
  assert.ok(h.jobs.has(second), "an earlier failure must not block later summaries");
});

test("overlapping summaries keep progress and completion attached to their meetings", async (t) => {
  const h = await mount(t);
  await h.click();
  const first = h.starts[0];
  await h.click();
  await h.completeTranscript(first);
  await h.click();
  const second = h.starts[1];
  await h.click();
  await h.completeTranscript(second);
  await h.emit("meetingSummaryToken", { meetingId: first, chunk: "first notes" });
  assert.equal(h.view("DetailView").props.liveSummary, "");
  await h.emit("meetingSummaryToken", { meetingId: second, chunk: "second notes" });
  assert.equal(h.view("DetailView").props.liveSummary, "second notes");
  await h.finishSummary(first);
  assert.equal(h.view("DetailView").props.meeting.id, second);
  assert.equal(h.view("DetailView").props.summarizing, true);
  await h.emit("meetingSummaryToken", { meetingId: first, chunk: "late old event" });
  assert.equal(h.view("DetailView").props.liveSummary, "second notes");
  await h.finishSummary(second);
  assert.equal(h.view("DetailView").props.summarizing, false);
  assert.equal(h.view("DetailView").props.meeting.summary, `Notes for ${second}`);
});

test("old notes finishing during recorder startup cannot enable a duplicate start", async (t) => {
  const h = await mount(t);
  await h.click();
  const first = h.starts[0];
  await h.click();
  await h.completeTranscript(first);
  const starting = deferred();
  const start = fixture.api.meetingStart;
  fixture.api.meetingStart = async (...args) => { await starting.promise; return start(...args); };
  await h.emit("meetingStartRequest");
  await h.finishSummary(first);
  assert.equal(h.pill().props.disabled, true);
  await h.emit("meetingToggle");
  await act(async () => starting.resolve());
  assert.equal(h.starts.length, 2);
  assert.equal(h.view("LiveView").props.meetingId, h.starts[1]);
});

test("delayed archive loading cannot block or overwrite the next recording", async (t) => {
  const h = await mount(t);
  await h.click();
  const tasks = deferred(), status = deferred();
  fixture.api.listTasks = () => tasks.promise;
  fixture.api.meetingWorkStatus = () => status.promise;
  await h.click();
  await h.click();
  const second = h.starts[1];
  await act(async () => {
    tasks.resolve([{ id: "old task" }]);
    status.reject(new Error("status unavailable"));
  });
  assert.equal(h.view("LiveView").props.meetingId, second);
  assert.deepEqual(h.view("LiveView").props.tasks, []);
  assert.match(h.pill().props.className, /live/);
});

test("a failed native stop keeps the stop control available", async (t) => {
  const h = await mount(t);
  await h.click();
  const stop = fixture.api.meetingStop;
  fixture.api.meetingStop = async () => { throw new Error("stop failed"); };
  await h.click();
  assert.match(h.pill().props.className, /live/);
  fixture.api.meetingStop = stop;
  await h.click();
  assert.equal(fixture.active, null);
});
