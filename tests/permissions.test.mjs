import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import ts from "typescript";
import React from "react";
import { act, create } from "react-test-renderer";

const modelUrl = new URL("../src/permissionsModel.ts", import.meta.url).href;
const componentUrl = new URL("../src/Permissions.tsx", import.meta.url).href;
const react = JSON.stringify(import.meta.resolve("react"));
const virtual = (source) => ({ url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true });
const hooks = registerHooks({
  resolve(specifier, context, next) {
    if (context.parentURL === componentUrl) {
      if (specifier === "./permissionsModel") return { url: modelUrl, shortCircuit: true };
      if (specifier === "@tauri-apps/api/core") return virtual("export const invoke = (...a) => globalThis.permTest.invoke(...a);");
      if (specifier === "./api") return virtual("export const api = { requestMicrophone: () => globalThis.permTest.invoke('request_microphone') };");
      if (specifier === "./sections") return virtual(`import { createElement } from ${react};
        export const PrefGroup = (p) => createElement("section", { title: p.title }, p.children);
        export const PrefCard = (p) => createElement("div", null, p.children);`);
      if (specifier === "./ui") return virtual(`import { createElement } from ${react};
        export const Icon = (p) => createElement("Icon", p);`);
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

// Minimal window + localStorage for the hook's timer and focus listener.
const listeners = new Map();
const store = new Map();
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
};
globalThis.window = {
  setInterval: () => 1,
  clearInterval() {},
  addEventListener: (type, fn) => listeners.set(type, [...(listeners.get(type) ?? []), fn]),
  removeEventListener: (type, fn) => listeners.set(type, (listeners.get(type) ?? []).filter((f) => f !== fn)),
  dispatchEvent: (e) => (listeners.get(e.type) ?? []).forEach((fn) => fn(e)),
};
let fixture;
globalThis.permTest = { invoke: (cmd, args) => fixture.invoke(cmd, args) };

const model = await import(modelUrl);
const { PermissionsPane, PermissionsStep, PermissionsCheck, useOpenPermissions } = await import(componentUrl);
hooks.deregister();

const {
  actionFor, needsReview, needsRelaunch, missingRequired, readKinds, addKinds, statusLabel, detailFor,
  defFor, PERMISSIONS, ASKED_KEY, REVIEWED_KEY, openPermissions,
} = model;

const all = (over = {}) => ["microphone", "accessibility", "screenRecording", "calendar", "automation", "notifications"]
  .map((kind) => ({ kind, access: "granted", ...(kind === "calendar" ? { enabled: true } : {}), ...over[kind] }));

/* ── pure decisions ── */

test("each status maps to the action the row offers", () => {
  const e = (kind, access, extra = {}) => ({ kind, access, ...extra });
  assert.equal(actionFor(e("microphone", "granted"), false), "none");
  assert.equal(actionFor(e("calendar", "granted", { enabled: false }), false), "enable");
  assert.equal(actionFor(e("calendar", "granted", { enabled: true }), false), "none");
  assert.equal(actionFor(e("microphone", "denied"), false), "settings");
  assert.equal(actionFor(e("calendar", "restricted"), false), "settings");
  assert.equal(actionFor(e("microphone", "notDetermined"), true), "allow", "the mic prompt reports its answer");
  assert.equal(actionFor(e("accessibility", "notDetermined"), false), "allow");
  assert.equal(actionFor(e("accessibility", "notDetermined"), true), "settings", "prompts once, then Settings");
  assert.equal(actionFor(e("screenRecording", "notDetermined"), true), "settings");
  assert.equal(actionFor(e("automation", "unknown"), false), "allow");
  assert.equal(actionFor(e("automation", "unknown"), true), "settings");
  assert.equal(actionFor(e("notifications", "unknown"), false), "settings");
  assert.equal(actionFor(e("screenRecording", "unsupported"), false), "none");
  assert.equal(actionFor(undefined, false), "none");
});

test("labels and descriptions explain what to do", () => {
  const cal = defFor("calendar");
  assert.deepEqual(statusLabel(cal, { kind: "calendar", access: "granted", enabled: true }), { text: "Allowed", tone: "ok" });
  assert.equal(statusLabel(cal, { kind: "calendar", access: "granted", enabled: false }).tone, "neutral");
  assert.equal(statusLabel(defFor("automation"), { kind: "automation", access: "unknown" }).text, "Asks on first use");
  assert.match(detailFor(cal, { kind: "calendar", access: "denied" }), /System Settings/);
  assert.match(detailFor(cal, { kind: "calendar", access: "granted", enabled: false }), /Use my calendar/);
  assert.match(defFor("accessibility").why, /who's on it/);
});

test("dictation's permissions come first and are the only required ones", () => {
  assert.deepEqual(PERMISSIONS.filter((p) => p.required).map((p) => p.kind), ["microphone", "accessibility"]);
  assert.deepEqual(PERMISSIONS.slice(0, 2).map((p) => p.kind), ["microphone", "accessibility"]);
  assert.ok(PERMISSIONS.every((p) => p.title && p.why));
  assert.ok(defFor("screenRecording").relaunch);
});

test("existing users are checked only for required permissions they haven't been shown", () => {
  const off = all({ accessibility: { access: "notDetermined" } });
  assert.equal(needsReview(false, [], off), false, "fresh installs see it in onboarding instead");
  assert.equal(needsReview(true, [], off), true);
  assert.equal(needsReview(true, ["accessibility"], off), false);
  assert.equal(needsReview(true, ["accessibility"], all({ accessibility: { access: "denied" }, microphone: { access: "denied" } })), true);
  assert.equal(needsReview(true, [], all({ screenRecording: { access: "denied" }, calendar: { access: "denied" } })), false, "optional ones never block");
  assert.equal(needsReview(true, [], all({ microphone: { access: "unknown" } })), false);
  assert.deepEqual(missingRequired(all({ microphone: { access: "restricted" } })), ["microphone"]);
});

test("relaunch is offered only after asking for a launch-applied permission that still reads as off", () => {
  const off = all({ screenRecording: { access: "notDetermined" } });
  assert.deepEqual(needsRelaunch(off, ["screenRecording"]), ["screenRecording"]);
  assert.deepEqual(needsRelaunch(off, []), []);
  assert.deepEqual(needsRelaunch(all(), ["screenRecording"]), []);
  assert.deepEqual(needsRelaunch(all({ accessibility: { access: "notDetermined" } }), ["accessibility"]), []);
});

test("asked/reviewed lists survive bad storage", () => {
  const s = new Map();
  const fake = { getItem: (k) => s.get(k) ?? null, setItem: (k, v) => s.set(k, v) };
  assert.deepEqual(readKinds(fake, "k"), []);
  s.set("k", "not json");
  assert.deepEqual(readKinds(fake, "k"), []);
  s.set("k", JSON.stringify(["calendar", "location"]));
  assert.deepEqual(readKinds(fake, "k"), ["calendar"]);
  assert.deepEqual(addKinds(fake, "k", ["calendar", "microphone"]), ["calendar", "microphone"]);
  const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
  assert.deepEqual(addKinds(broken, "k", ["microphone"]), ["microphone"]);
  assert.deepEqual(readKinds(undefined, "k"), []);
});

/* ── components ── */

const text = (node) => typeof node === "string" ? node : (node.children ?? []).map(text).join("");
const button = (root, label) => root.find((n) => n.type === "button" && text(n).includes(label));
const buttons = (root, label) => root.findAll((n) => n.type === "button" && text(n).includes(label));
const rows = (root) => root.findAll((n) => n.props?.["data-permission"]).map((n) => n.props["data-permission"]);
const settle = () => act(() => new Promise(setImmediate));

function harness(entries) {
  store.clear();
  listeners.clear();
  const calls = [];
  fixture = {
    entries,
    calls,
    invoke: async (cmd, args) => {
      calls.push(args ? [cmd, args] : [cmd]);
      if (cmd === "permissions_all") return fixture.entries;
      if (cmd === "permissions_request") return "notDetermined";
    },
  };
  return fixture;
}

test("onboarding step asks, then falls back to System Settings, and unlocks Continue live", async () => {
  const f = harness(all({ accessibility: { access: "notDetermined" }, screenRecording: { access: "notDetermined" } }));
  let next = 0;
  let r;
  await act(async () => { r = create(React.createElement(PermissionsStep, { onNext: () => next++, onError() {} })); });
  await settle();
  const root = r.root;
  assert.deepEqual(rows(root), ["microphone", "accessibility", "screenRecording", "calendar"], "Automation and Notifications live in Settings only");
  assert.equal(button(root, "Continue").props.disabled, true);
  assert.equal(buttons(root, "Optional").length, 0);
  assert.equal(root.findAll((n) => n.props?.className === "badge neutral" && text(n) === "Optional").length, 2);
  assert.deepEqual(JSON.parse(store.get(REVIEWED_KEY)), ["microphone", "accessibility"]);

  const accessibilityRow = () => root.find((n) => n.props?.["data-permission"] === "accessibility");
  await act(async () => button(accessibilityRow(), "Allow").props.onClick());
  assert.deepEqual(f.calls.find((c) => c[0] === "permissions_request"), ["permissions_request", { kind: "accessibility" }]);
  assert.deepEqual(JSON.parse(store.get(ASKED_KEY)), ["accessibility"]);
  await act(async () => button(accessibilityRow(), "Open Settings").props.onClick());
  assert.deepEqual(f.calls.at(-2), ["permissions_open_settings", { kind: "accessibility" }]);

  // Granted in System Settings; the window regains focus.
  f.entries = all({ screenRecording: { access: "notDetermined" } });
  await act(async () => window.dispatchEvent({ type: "focus" }));
  await settle();
  assert.equal(button(root, "Continue").props.disabled, false);
  assert.equal(buttons(root, "Set up later").length, 0);
  button(root, "Continue").props.onClick();
  assert.equal(next, 1);
  act(() => r.unmount());
});

test("Screen Recording shows a relaunch prompt once asked", async () => {
  const f = harness(all({ screenRecording: { access: "notDetermined" } }));
  let r;
  await act(async () => { r = create(React.createElement(PermissionsStep, { onNext() {}, onError() {} })); });
  await settle();
  assert.equal(buttons(r.root, "Relaunch").length, 0);
  const screen = r.root.find((n) => n.props?.["data-permission"] === "screenRecording");
  await act(async () => button(screen, "Allow").props.onClick());
  await settle();
  await act(async () => button(r.root, "Relaunch unsound").props.onClick());
  assert.deepEqual(f.calls.at(-1), ["relaunch_app"]);
  act(() => r.unmount());
});

test("settings page lists every permission with its status and turns calendar on", async () => {
  const f = harness(all({
    calendar: { access: "granted", enabled: false },
    automation: { access: "unknown" },
    notifications: { access: "unknown" },
    screenRecording: { access: "denied" },
  }));
  let r;
  await act(async () => { r = create(React.createElement(PermissionsPane, { onError() {} })); });
  await settle();
  const root = r.root;
  assert.deepEqual(rows(root), PERMISSIONS.map((p) => p.kind));
  const badges = root.findAll((n) => typeof n.props?.className === "string" && n.props.className.startsWith("badge")).map(text);
  assert.deepEqual(badges, ["Allowed", "Allowed", "Off", "Allowed · not in use", "Asks on first use", "Set in System Settings"]);
  const row = (kind) => root.find((n) => n.props?.["data-permission"] === kind);
  await act(async () => button(row("calendar"), "Turn on").props.onClick());
  assert.ok(f.calls.some((c) => c[0] === "permissions_request" && c[1].kind === "calendar"));
  await act(async () => button(row("screenRecording"), "Open Settings").props.onClick());
  assert.ok(f.calls.some((c) => c[0] === "permissions_open_settings" && c[1].kind === "screenRecording"));
  await act(async () => button(row("microphone"), "Test microphone").props.onClick());
  assert.ok(f.calls.some((c) => c[0] === "request_microphone"));
  assert.equal(text(row("microphone")).includes("Working"), true);
  act(() => r.unmount());
});

test("launch check appears once per missing required permission", async () => {
  harness(all({ accessibility: { access: "notDetermined" } }));
  let r;
  await act(async () => { r = create(React.createElement(PermissionsCheck)); });
  await settle();
  assert.equal(text(r.root.findByProps({ id: "perm-check-title" })), "unsound needs access");
  await act(async () => button(r.root, "Not now").props.onClick());
  assert.equal(r.toJSON(), null);
  assert.deepEqual(JSON.parse(store.get(REVIEWED_KEY)), ["accessibility"]);
  act(() => r.unmount());

  // Next launch: same permission still off, already shown.
  await act(async () => { r = create(React.createElement(PermissionsCheck)); });
  await settle();
  assert.equal(r.toJSON(), null);
  act(() => r.unmount());

  harness(all());
  await act(async () => { r = create(React.createElement(PermissionsCheck)); });
  await settle();
  assert.equal(r.toJSON(), null, "nothing missing, nothing shown");
  act(() => r.unmount());
});

test("openPermissions reaches every listener", async () => {
  listeners.clear();
  let opened = 0;
  function Probe() { useOpenPermissions(React.useCallback(() => opened++, [])); return null; }
  let r;
  await act(async () => { r = create(React.createElement(Probe)); });
  openPermissions();
  assert.equal(opened, 1);
  act(() => r.unmount());
  openPermissions();
  assert.equal(opened, 1);
});
