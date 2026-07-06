import { invoke } from "@tauri-apps/api/core";
import { emit, listen, UnlistenFn } from "@tauri-apps/api/event";
import type { Take } from "./HistoryDrawer";

export type OverlayState = "recording" | "processing" | "hidden";

export type ModelKind = "stt" | "llm";

export interface ModelInfo {
  id: string;
  name: string;
  kind: ModelKind;
  url: string;
  filename: string;
  sizeBytes: number;
  description: string;
  languages: string;
  recommended: boolean;
  custom: boolean;
  downloaded: boolean;
}

export interface PermissionStatus {
  accessibility: boolean;
}

export interface RecordingResult {
  durationSecs: number;
  sampleCount: number;
}

export interface DownloadProgress {
  id: string;
  downloaded: number;
  total: number;
}

export interface DownloadDone {
  id: string;
  ok: boolean;
  error: string | null;
}

export interface DictEntry {
  from: string;
  to: string;
}

export interface Style {
  id: string;
  name: string;
  notes: string;
  lowercase: boolean;
  samples: string[];
}

export interface Settings {
  handsFree: string[];
  pushToTalk: string[];
  micDevice: string;
  styles: Style[];
  defaultStyle: string;
  dictionary: DictEntry[];
}

/// One utterance in a meeting transcript. `speaker` is free-form so "me"/"them"
/// extends to named or numbered participants later; `source` is the channel it
/// came from (mic vs. system audio).
export interface Segment {
  id?: number;
  speaker: string;
  source: string;
  startMs: number;
  endMs: number;
  text: string;
}

export interface Meeting {
  id: string;
  title: string;
  startedAt: string;
  endedAt?: string | null;
  summary: string;
  notes: string;
  sttModel: string;
  llmModel: string;
  lang?: string | null;
  segments: Segment[];
  segmentCount: number;
}

export interface SearchHit {
  meetingId: string;
  title: string;
  startedAt: string;
  snippet: string;
}

export const api = {
  listModels: () => invoke<ModelInfo[]>("list_models"),
  downloadModel: (id: string) => invoke<void>("download_model", { id }),
  deleteModel: (id: string) => invoke<void>("delete_model", { id }),
  addCustomModel: (name: string, kind: ModelKind, url: string) =>
    invoke<ModelInfo>("add_custom_model", { name, kind, url }),
  startRecording: () => invoke<void>("start_recording"),
  stopRecording: () => invoke<RecordingResult>("stop_recording"),
  transcribe: (modelId: string, language?: string, translate = false) =>
    invoke<string>("transcribe", { modelId, language: language ?? null, translate }),
  transcribeFile: (path: string, modelId: string, language?: string, translate = false) =>
    invoke<string>("transcribe_file", { path, modelId, language: language ?? null, translate }),
  cleanupText: (
    modelId: string,
    text: string,
    prompt?: string,
    styleId?: string,
    targetLang?: string,
    transliterate = false,
  ) =>
    invoke<string>("cleanup_text", {
      modelId,
      text,
      prompt: prompt ?? null,
      styleId: styleId ?? null,
      targetLang: targetLang ?? null,
      transliterate,
    }),
  defaultCleanupPrompt: () => invoke<string>("default_cleanup_prompt"),
  getSettings: () => invoke<Settings>("get_settings"),
  setShortcuts: (handsFree: string[], pushToTalk: string[]) =>
    invoke<void>("set_shortcuts", { handsFree, pushToTalk }),
  setStyles: (styles: Style[], defaultStyle: string) =>
    invoke<void>("set_styles", { styles, defaultStyle }),
  deliverText: (text: string) => invoke<string>("deliver_text", { text }),
  addCorrection: (from: string, to: string) => invoke<void>("add_correction", { from, to }),
  setDictionary: (entries: DictEntry[]) => invoke<void>("set_dictionary", { entries }),
  permissionStatus: () => invoke<PermissionStatus>("permission_status"),
  startShortcutCapture: () => invoke<boolean>("start_shortcut_capture"),
  cancelShortcutCapture: () => invoke<void>("cancel_shortcut_capture"),
  requestAccessibility: () => invoke<boolean>("request_accessibility"),
  requestMicrophone: () => invoke<void>("request_microphone"),
  listMicrophones: () => invoke<string[]>("list_microphones"),
  setMicrophone: (device: string) => invoke<void>("set_microphone", { device }),
  setOverlay: (visible: boolean) => invoke<void>("set_overlay", { visible }),
  emitOverlayState: (state: OverlayState) => emit("overlay-state", state),

  // History — now stored in SQLite (was localStorage).
  listTakes: () => invoke<Take[]>("list_takes"),
  saveTake: (take: Take) => invoke<void>("save_take", { take }),
  deleteTake: (id: string) => invoke<void>("delete_take", { id }),
  clearTakes: () => invoke<void>("clear_takes"),
  importTakes: (takes: Take[]) => invoke<number>("import_takes", { takes }),

  // Meetings.
  createMeeting: (meeting: Meeting) => invoke<void>("create_meeting", { meeting }),
  addMeetingSegments: (meetingId: string, segments: Segment[]) =>
    invoke<void>("add_meeting_segments", { meetingId, segments }),
  endMeeting: (id: string, endedAt: string, summary: string, title?: string) =>
    invoke<void>("end_meeting", { id, endedAt, summary, title: title ?? null }),
  updateMeetingNotes: (id: string, notes: string) =>
    invoke<void>("update_meeting_notes", { id, notes }),
  setMeetingSummary: (id: string, summary: string) =>
    invoke<void>("set_meeting_summary", { id, summary }),
  renameMeeting: (id: string, title: string) => invoke<void>("rename_meeting", { id, title }),
  deleteMeeting: (id: string) => invoke<void>("delete_meeting", { id }),
  listMeetings: () => invoke<Meeting[]>("list_meetings"),
  getMeeting: (id: string) => invoke<Meeting | null>("get_meeting", { id }),
  transcribeMeeting: (meetingId: string, modelId: string, language?: string) =>
    invoke<Meeting>("transcribe_meeting", { meetingId, modelId, language: language ?? null }),
  summarizeMeeting: (meetingId: string, modelId: string) =>
    invoke<string>("summarize_meeting", { meetingId, modelId }),
  askMeeting: (meetingId: string, modelId: string, question: string) =>
    invoke<string>("ask_meeting", { meetingId, modelId, question }),
  searchMeetings: (query: string) => invoke<SearchHit[]>("search_meetings", { query }),

  // System-audio capture (ScreenCaptureKit; macOS 13+).
  systemAudioSupported: () => invoke<boolean>("system_audio_supported"),
  startSystemCapture: () => invoke<void>("start_system_capture"),
  stopSystemCapture: () => invoke<RecordingResult>("stop_system_capture"),
  saveSystemCaptureWav: () =>
    invoke<{ path: string; sampleCount: number; durationSecs: number; peak: number; rms: number }>(
      "save_system_capture_wav",
    ),
  transcribeSystemCapture: (modelId: string, language?: string) =>
    invoke<string>("transcribe_system_capture", { modelId, language: language ?? null }),
};

export const on = {
  audioLevel: (cb: (rms: number) => void): Promise<UnlistenFn> =>
    listen<number>("audio-level", (e) => cb(e.payload)),
  downloadProgress: (cb: (p: DownloadProgress) => void): Promise<UnlistenFn> =>
    listen<DownloadProgress>("model-download-progress", (e) => cb(e.payload)),
  downloadDone: (cb: (d: DownloadDone) => void): Promise<UnlistenFn> =>
    listen<DownloadDone>("model-download-done", (e) => cb(e.payload)),
  llmToken: (cb: (chunk: string) => void): Promise<UnlistenFn> =>
    listen<string>("llm-token", (e) => cb(e.payload)),
  meetingSummaryToken: (cb: (chunk: string) => void): Promise<UnlistenFn> =>
    listen<string>("meeting-summary-token", (e) => cb(e.payload)),
  meetingAnswerToken: (cb: (chunk: string) => void): Promise<UnlistenFn> =>
    listen<string>("meeting-answer-token", (e) => cb(e.payload)),
  hotkeyToggle: (cb: () => void): Promise<UnlistenFn> =>
    listen<void>("hotkey-toggle", () => cb()),
  pttDown: (cb: () => void): Promise<UnlistenFn> => listen<void>("ptt-down", () => cb()),
  pttUp: (cb: () => void): Promise<UnlistenFn> => listen<void>("ptt-up", () => cb()),
  pttCancel: (cb: () => void): Promise<UnlistenFn> => listen<void>("ptt-cancel", () => cb()),
  captureUpdate: (cb: (combo: string) => void): Promise<UnlistenFn> =>
    listen<{ combo: string }>("capture-update", (e) => cb(e.payload.combo)),
  captureCommit: (cb: (combo: string) => void): Promise<UnlistenFn> =>
    listen<{ combo: string }>("capture-commit", (e) => cb(e.payload.combo)),
  captureCancel: (cb: () => void): Promise<UnlistenFn> =>
    listen<void>("capture-cancel", () => cb()),
  overlayState: (cb: (state: OverlayState) => void): Promise<UnlistenFn> =>
    listen<OverlayState>("overlay-state", (e) => cb(e.payload)),
  settingsChanged: (cb: () => void): Promise<UnlistenFn> =>
    listen<void>("settings-changed", () => cb()),
  fileInfo: (cb: (info: { sizeBytes: number; durationSecs: number }) => void): Promise<UnlistenFn> =>
    listen<{ sizeBytes: number; durationSecs: number }>("file-info", (e) => cb(e.payload)),
};

export function formatDuration(secs: number): string {
  const s = Math.round(secs);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}:${String(r).padStart(2, "0")}` : `${r}s`;
}

export function formatShortcut(shortcut: string): string {
  if (!shortcut) return "disabled";
  return shortcut
    .split("+")
    .map((part) => {
      switch (part.toLowerCase()) {
        case "cmd":
        case "super":
        case "command":
          return "⌘";
        case "shift":
          return "⇧";
        case "alt":
        case "option":
          return "⌥";
        case "ctrl":
        case "control":
          return "⌃";
        case "fn":
          return "fn";
        case "space":
          return "Space";
        case "backspace":
          return "⌫";
        case "delete":
          return "⌦";
        case "enter":
          return "↩";
        case "tab":
          return "⇥";
        case "home":
          return "Home";
        case "end":
          return "End";
        case "up":
          return "↑";
        case "down":
          return "↓";
        case "left":
          return "←";
        case "right":
          return "→";
        default:
          return part.toUpperCase();
      }
    })
    .join(" ");
}

// Common Whisper languages; "" means auto-detect. Whisper supports ~99 —
// this is the frequently-used subset, alphabetical after Auto.
export const LANGUAGES: { code: string; name: string }[] = [
  { code: "", name: "Auto-detect" },
  { code: "ar", name: "Arabic" },
  { code: "zh", name: "Chinese" },
  { code: "nl", name: "Dutch" },
  { code: "en", name: "English" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "hi", name: "Hindi" },
  { code: "it", name: "Italian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "pl", name: "Polish" },
  { code: "pa", name: "Punjabi" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "es", name: "Spanish" },
  { code: "sv", name: "Swedish" },
  { code: "tr", name: "Turkish" },
  { code: "uk", name: "Ukrainian" },
  { code: "ur", name: "Urdu" },
  { code: "vi", name: "Vietnamese" },
];

export function formatBytes(n: number): string {
  if (!n) return "?";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}
