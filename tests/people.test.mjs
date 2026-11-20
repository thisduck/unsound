import { test } from "node:test";
import assert from "node:assert/strict";
import { meetingPeople, onCall, remoteCountHint, speakerSuggestions } from "../src/meetings/people.ts";

const person = (name, me = false) => ({ name, me, app: "Zoom", firstMs: 0, lastMs: 0 });

test("the call's roster comes first, without you, deduplicated with invitees", () => {
  const m = {
    participants: { people: [person("Adnan", true), person("Priya Raman"), person("Marcus")], solo: [] },
    attendees: ["priya raman", "Dana", "lee@example.com"],
  };
  assert.deepEqual(onCall(m), ["Priya Raman", "Marcus"]);
  assert.deepEqual(meetingPeople(m), ["Priya Raman", "Marcus", "Dana", "lee@example.com"]);
});

test("suggestions skip names already given and email addresses", () => {
  const m = {
    participants: { people: [person("Priya Raman"), person("Marcus")], solo: [] },
    attendees: ["Dana", "lee@example.com"],
    speakerNames: { "them:0": "priya raman", me: "Adnan" },
  };
  assert.deepEqual(speakerSuggestions(m), ["Marcus", "Dana"]);
  assert.deepEqual(speakerSuggestions({}), []);
});

test("count hint prefers the call over the calendar and stays in range", () => {
  assert.deepEqual(remoteCountHint({ participants: { people: [person("A", true), person("B"), person("C")], solo: [] }, attendees: ["X"] }), { count: 2, from: "call" });
  assert.deepEqual(remoteCountHint({ attendees: ["X", "Y", "Y"] }), { count: 2, from: "calendar" });
  assert.equal(remoteCountHint({}), null);
  assert.equal(remoteCountHint({ attendees: Array.from({ length: 9 }, (_, i) => `P${i}`) }), null);
});
