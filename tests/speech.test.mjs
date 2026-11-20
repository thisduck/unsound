import { test } from "node:test";
import assert from "node:assert/strict";
import { fallbackWhisper, languageNote, migrateSttId, supportedLanguageNames } from "../src/speech.ts";

const model = (id, extra = {}) => ({
  id,
  name: id,
  kind: "stt",
  url: "",
  filename: "",
  sizeBytes: 0,
  description: "",
  languages: "",
  recommended: false,
  custom: false,
  downloaded: true,
  ...extra,
});
const v2 = model("parakeet-tdt-0.6b-v2", { name: "Parakeet v2", engine: "parakeet", supportedLanguages: ["en"] });
const v3 = model("parakeet-tdt-0.6b-v3", {
  name: "Parakeet v3",
  engine: "parakeet",
  supportedLanguages: ["bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "uk"],
});
const turbo = model("whisper-large-v3-turbo", { name: "Whisper Turbo", engine: "whisper" });
const tiny = model("whisper-tiny", { name: "Whisper Tiny", engine: "whisper" });

test("the removed sherpa Parakeet id becomes the new v3", () => {
  assert.equal(migrateSttId("parakeet-tdt-v3"), "parakeet-tdt-0.6b-v3");
  assert.equal(migrateSttId("whisper-large-v3-turbo"), "whisper-large-v3-turbo");
  assert.equal(migrateSttId(""), "");
});

test("the fallback is the most accurate downloaded multilingual Whisper", () => {
  assert.equal(fallbackWhisper([v3, tiny, turbo])?.id, "whisper-large-v3-turbo");
  assert.equal(fallbackWhisper([v3, { ...turbo, downloaded: false }, tiny])?.id, "whisper-tiny");
  assert.equal(fallbackWhisper([v3, model("whisper-small-en")]), undefined);
});

test("no note when the model simply transcribes the language", () => {
  assert.equal(languageNote(turbo, "hi", [turbo]), null);
  assert.equal(languageNote(v3, "de", [v3]), null);
  assert.equal(languageNote(v2, "en", [v2]), null);
  assert.equal(languageNote(v3, "", [v3, turbo]), null);
  assert.equal(languageNote(undefined, "hi", []), null);
});

test("an unsupported language goes to Whisper, or is refused without one", () => {
  assert.deepEqual(languageNote(v3, "hi", [v3, turbo]), {
    tone: "info",
    text: "Parakeet v3 doesn't do Hindi, so Whisper Turbo transcribes it.",
  });
  const refused = languageNote(v2, "fr", [v2]);
  assert.equal(refused.tone, "warn");
  assert.match(refused.text, /^Parakeet v2 can't transcribe French\./);
});

test("auto-detect without Whisper says what Parakeet assumes", () => {
  assert.match(languageNote(v2, "", [v2]).text, /treats all speech as English\.$/);
  assert.match(languageNote(v3, "", [v3]).text, /treats all speech as one of its 25 languages\.$/);
});

test("model cards list the supported languages by name", () => {
  assert.equal(supportedLanguageNames(v2), "English");
  assert.equal(supportedLanguageNames(turbo), "");
  const names = supportedLanguageNames(v3).split(", ");
  assert.equal(names.length, 25);
  assert.ok(names.includes("Maltese") && names.includes("Ukrainian"));
});
