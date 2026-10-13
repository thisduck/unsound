import { test } from "node:test";
import assert from "node:assert/strict";
import { createNotesDraft } from "../src/meetings/notesDraft.ts";

const storage = () => {
  const values = new Map();
  return { getItem: (k) => values.get(k) ?? null, setItem: (k,v) => values.set(k,v), removeItem: (k) => values.delete(k) };
};

test("failed save preserves draft across editor recreation and retry", async () => {
  const disk = storage();
  const states = [];
  const first = createNotesDraft("failure",disk,async () => { throw new Error("disk full"); },s => states.push(s));
  first.edit("important notes");
  await first.flush();
  assert.match(states.at(-1).error,/disk full/);
  let saved;
  const reopened = createNotesDraft("failure",disk,async (_id,text) => { saved=text; },() => {});
  assert.equal(reopened.restored,"important notes");
  await reopened.flush();
  assert.equal(saved,"important notes");
  assert.equal(disk.getItem("unsound.notes-draft.failure"),null);
});

test("live-to-archive handoff serializes writes and preserves the newer draft", async () => {
  const disk=storage();
  let release;
  const gate=new Promise(resolve => { release=resolve; });
  const saved=[];
  const first=createNotesDraft("handoff",disk,async (_id,text) => { await gate; saved.push(text); },() => {});
  first.edit("older");
  const pending=first.flush(); // unmount flush while storage is stalled
  const next=createNotesDraft("handoff",disk,async (_id,text) => { saved.push(text); },() => {});
  next.edit("newer");
  const later=next.flush();
  assert.deepEqual(saved,[]);
  release();
  await pending;
  assert.equal(disk.getItem("unsound.notes-draft.handoff"),"newer");
  await later;
  assert.deepEqual(saved,["older","newer"]);
  assert.equal(disk.getItem("unsound.notes-draft.handoff"),null);
});

test("an edit during a pending save is not acknowledged by the older result", async () => {
  const disk=storage();
  let release;
  const gate=new Promise(resolve => { release=resolve; });
  const d=createNotesDraft("typing",disk,async () => gate,() => {});
  d.edit("first");
  const pending=d.flush();
  d.edit("second");
  release();
  await pending;
  assert.equal(disk.getItem("unsound.notes-draft.typing"),"second");
  await d.flush();
  assert.equal(disk.getItem("unsound.notes-draft.typing"),null);
});

test("draft storage failure is visible and does not prevent a database save", async () => {
  const states=[];
  let saved;
  const d=createNotesDraft("quota", {getItem:()=>null,setItem:()=>{throw new Error("quota");},removeItem:()=>{}},
    async (_id,text)=>{saved=text;},s=>states.push(s));
  d.edit("keep me");
  assert.match(states.at(-1).error,/local draft could not be saved/);
  await d.flush();
  assert.equal(saved,"keep me");
  assert.equal(states.at(-1).error,null);
});
