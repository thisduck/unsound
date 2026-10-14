import { invoke } from "@tauri-apps/api/core";
import { emit, listen, UnlistenFn } from "@tauri-apps/api/event";
import type { Take } from "./HistoryDrawer";

export type OverlayState = "recording" | "processing" | "command" | "hidden";

export type ModelKind = "stt" | "llm" | "diarize" | "vad";

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
  /// Multi-file bundle models (e.g. Moonshine) list their file URLs; empty for
  /// ordinary single-file models.
  files?: string[];
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

export interface JiraConfig {
  site: string;
  email: string;
  defaultProject: string;
  defaultIssueType: string;
}

export interface McpServer {
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  url: string;
  headers: Record<string, string>;
  enabled: boolean;
}

export interface ToolInfo {
  server: string;
  name: string;
  description: string;
  input_schema: unknown;
}

/// A tool described in the user's own words (see settings.rs ToolGuide).
export interface ToolGuide {
  id: string;
  name: string;
  when: string;
  site: string;
  guidance: string;
  /// "browser" | "auto" | "draft"
  mode: string;
  leaveFinalStep: boolean;
  enabled: boolean;
}

export interface AssistantSettings {
  enabled: boolean;
  wakeWord: string;
  listenForWakeWord: boolean;
  /// "local" | "claude" | "codex" | "anthropic" | "openai" | "compatible"
  backend: string;
  localModel: string;
  cloudModel: string;
  cliModel: string;
  endpointUrl: string;
  endpointModel: string;
  autoRun: string[];
  jira: JiraConfig;
  mcpServers: McpServer[];
  claudeCode: boolean;
  claudeAllowedTools: string[];
  claudeModel: string;
  guides: ToolGuide[];
  aboutMe: string;
  /// "chrome" | "local" | "off"
  browser: string;
  browserServer: McpServer;
}

export interface MeetingSettings {
  detectMeetings: boolean;
  useCalendar: boolean;
  embeddingModel: string;
  summaryModel: string;
}

export interface Settings {
  handsFree: string[];
  pushToTalk: string[];
  micDevice: string;
  styles: Style[];
  defaultStyle: string;
  dictionary: DictEntry[];
  commandKeys: string[];
  assistant: AssistantSettings;
  meetings: MeetingSettings;
}

export type CalAccess = "notDetermined" | "denied" | "granted";

export interface CalEvent {
  title: string;
  start: string;
  end: string;
  attendees: string[];
  organizer?: string | null;
  location: string;
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

export interface Mark {
  id: number;
  atMs: number;
  label: string;
}

export interface ActionItem {
  id: number;
  meetingId?: string | null;
  text: string;
  owner: string;
  due: string;
  atMs?: number | null;
  done: boolean;
  manual: boolean;
  createdAt: string;
  meetingTitle: string;
}

export interface Meeting {
  id: string;
  title: string;
  startedAt: string;
  endedAt?: string | null;
  summary: string;
  summaryStale?: boolean;
  notes: string;
  sttModel: string;
  llmModel: string;
  lang?: string | null;
  segments: Segment[];
  segmentCount: number;
  /// Per-meeting speaker display-name overrides (speaker key → name).
  speakerNames?: Record<string, string>;
  attendees?: string[];
  marks?: Mark[];
  actionItems?: ActionItem[];
  hasAudio?: boolean;
}

export interface MeetingWork {
  meetingId: string;
  state: "recording" | "finishing" | "paused" | "error" | "complete";
  error: string;
  warning: string;
  micSamples: number;
  systemSamples: number;
  pending: number;
  recording: boolean;
}

export interface SearchHit {
  meetingId: string;
  title: string;
  startedAt: string;
  snippet: string;
  /// "transcript" | "meeting"
  kind: string;
  segmentId?: number | null;
  startMs?: number | null;
  speaker?: string | null;
  speakerName?: string | null;
}

export type TaskAction =
  | "email"
  | "jira_issue"
  | "reminder"
  | "calendar_event"
  | "slack_message"
  | "note"
  | "text"
  | "browser"
  | "delegate";

export type TaskStatus =
  | "heard"
  | "planning"
  | "drafting"
  | "waiting"
  | "ready"
  | "running"
  | "done"
  | "failed"
  | "dismissed";

/// Something the user asked the assistant to do.
export interface Task {
  id: string;
  meetingId?: string | null;
  createdAt: string;
  updatedAt: string;
  atMs?: number | null;
  utterance: string;
  trigger: string;
  action: TaskAction | "";
  runWhen: string;
  status: TaskStatus;
  instruction: string;
  spanStartMs?: number | null;
  spanEndMs?: number | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  draft: any;
  result: { message?: string; url?: string } | null;
  error: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  log: any[];
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
  warmStt: (modelId: string) => invoke<void>("warm_stt", { modelId }),
  warmLlm: (
    modelId: string,
    prompt?: string,
    styleId?: string,
    targetLang?: string,
    transliterate = false,
  ) =>
    invoke<void>("warm_llm", {
      modelId,
      prompt: prompt ?? null,
      styleId: styleId ?? null,
      targetLang: targetLang ?? null,
      transliterate,
    }),
  startLiveDictation: (modelId: string, language?: string) =>
    invoke<void>("start_live_dictation", { modelId, language: language ?? null }),
  stopLiveDictation: () => invoke<void>("stop_live_dictation"),
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
  updateMeetingNotes: (id: string, notes: string) =>
    invoke<void>("update_meeting_notes", { id, notes }),
  setSpeakerName: (meetingId: string, speaker: string, name: string, remember = true) =>
    invoke<boolean>("set_speaker_name", { meetingId, speaker, name, remember }),
  correctMeetingSpeaker: (meetingId: string, segmentId: number, target: string | null, name: string | null, wholeGroup = false) =>
    invoke<Meeting>("correct_meeting_speaker", { meetingId, segmentId, target, name, wholeGroup }),
  setMeetingSummary: (id: string, summary: string) =>
    invoke<void>("set_meeting_summary", { id, summary }),
  updateSegmentText: (segmentId: number, text: string) =>
    invoke<void>("update_segment_text", { segmentId, text }),
  replaceInMeeting: (meetingId: string, from: string, to: string, remember: boolean) =>
    invoke<number>("replace_in_meeting", { meetingId, from, to, remember }),
  renameMeeting: (id: string, title: string) => invoke<void>("rename_meeting", { id, title }),
  deleteMeeting: (id: string) => invoke<void>("delete_meeting", { id }),
  listMeetings: () => invoke<Meeting[]>("list_meetings"),
  getMeeting: (id: string) => invoke<Meeting | null>("get_meeting", { id }),
  summarizeMeeting: (meetingId: string, modelId: string) =>
    invoke<Meeting>("summarize_meeting", { meetingId, modelId }),
  askMeeting: (meetingId: string, modelId: string, question: string) =>
    invoke<string>("ask_meeting", { meetingId, modelId, question }),
  askMeetings: (modelId: string, question: string) =>
    invoke<string>("ask_meetings", { modelId, question }),
  searchMeetings: (query: string) => invoke<SearchHit[]>("search_meetings", { query }),
  meetingStart: (
    meetingId: string,
    modelId: string,
    language?: string,
    llmModelId?: string,
    embeddingModelId?: string,
  ) =>
    invoke<Meeting | null>("meeting_start", {
      meetingId,
      modelId,
      language: language ?? null,
      llmModelId: llmModelId ?? null,
      embeddingModelId: embeddingModelId ?? null,
    }),
  meetingWorkStatus: () => invoke<MeetingWork[]>("meeting_work_status"),
  meetingPause: (id: string) => invoke<void>("meeting_pause", { id }),
  meetingRetry: (id: string) => invoke<void>("meeting_retry", { id }),
  meetingStop: () => invoke<Meeting | null>("meeting_stop"),
  diarizeMeeting: (meetingId: string, embeddingModelId?: string, numSpeakers?: number) =>
    invoke<Meeting>("diarize_meeting", {
      meetingId,
      embeddingModelId: embeddingModelId ?? null,
      numSpeakers: numSpeakers ?? null,
    }),
  meetingClip: (meetingId: string, startMs: number, endMs: number) =>
    invoke<ArrayBuffer>("meeting_clip", { meetingId, startMs, endMs }),
  addMark: (meetingId: string, label = "", atMs?: number) =>
    invoke<Mark>("add_mark", { meetingId, atMs: atMs ?? null, label }),
  deleteMark: (id: number) => invoke<void>("delete_mark", { id }),
  openActionItems: () => invoke<ActionItem[]>("open_action_items"),
  saveActionItem: (item: ActionItem) => invoke<ActionItem>("save_action_item", { item }),
  deleteActionItem: (id: number) => invoke<void>("delete_action_item", { id }),

  // Assistant tasks.
  listTasks: (meetingId?: string) => invoke<Task[]>("list_tasks", { meetingId: meetingId ?? null }),
  addTask: (text: string, meetingId?: string) =>
    invoke<Task | null>("add_task", { meetingId: meetingId ?? null, text }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  approveTask: (id: string, draft?: any) => invoke<Task>("approve_task", { id, draft: draft ?? null }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  updateTaskDraft: (id: string, draft: any, action?: string) =>
    invoke<Task>("update_task_draft", { id, draft, action: action ?? null }),
  retryTask: (id: string, utterance?: string) =>
    invoke<Task>("retry_task", { id, utterance: utterance ?? null }),
  redraftTask: (id: string) => invoke<Task>("redraft_task", { id }),
  dismissTask: (id: string) => invoke<Task>("dismiss_task", { id }),
  deleteTask: (id: string) => invoke<void>("delete_task", { id }),
  commandBegin: () => invoke<void>("command_begin"),
  commandEnd: () => invoke<void>("command_end"),
  assistantStatus: () =>
    invoke<{ ok: boolean; model?: string; error?: string; cloud?: boolean }>("assistant_status"),
  listCloudModels: (provider: string, base?: string) =>
    invoke<string[]>("list_cloud_models", { provider, base: base ?? null }),
  subscriptionClis: () => invoke<{ claude: string | null; codex: string | null }>("subscription_clis"),
  setAssistantSettings: (assistant: AssistantSettings) =>
    invoke<void>("set_assistant_settings", { assistant }),
  setMeetingSettings: (meetings: MeetingSettings) =>
    invoke<void>("set_meeting_settings", { meetings }),
  setCommandKeys: (keys: string[]) => invoke<void>("set_command_keys", { keys }),
  setSecret: (key: string, value: string) => invoke<void>("set_secret", { key, value }),
  hasSecret: (key: string) => invoke<boolean>("has_secret", { key }),
  jiraTest: () =>
    invoke<{ user: string; projects: { key: string; name: string }[] }>("jira_test"),
  mcpProbe: (server: McpServer) => invoke<ToolInfo[]>("mcp_probe", { server }),
  mcpParse: (json: string) => invoke<McpServer[]>("mcp_parse", { json }),
  claudeCodePath: () => invoke<string | null>("claude_code_path"),
  browserTest: () => invoke<string>("browser_test"),
  listVoices: () => invoke<[string, number][]>("list_voices"),
  deleteVoice: (name: string) => invoke<void>("delete_voice", { name }),
  calendarAccess: () => invoke<CalAccess>("calendar_access"),
  calendarRequest: () => invoke<boolean>("calendar_request"),
  calendarCurrent: () => invoke<CalEvent | null>("calendar_current"),
  hideMeetingPrompt: () => invoke<void>("hide_meeting_prompt"),

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
  meetingSegments: (cb: (p: { meetingId: string; segments: Segment[] }) => void): Promise<UnlistenFn> =>
    listen<{ meetingId: string; segments: Segment[] }>("meeting-segments", (e) => cb(e.payload)),
  meetingPartial: (cb: (p: { meetingId: string; speaker: string; text: string }) => void): Promise<UnlistenFn> =>
    listen<{ meetingId: string; speaker: string; text: string }>("meeting-partial", (e) => cb(e.payload)),
  meetingToggle: (cb: () => void): Promise<UnlistenFn> =>
    listen<void>("meeting-toggle", () => cb()),
  meetingsAnswerToken: (cb: (chunk: string) => void): Promise<UnlistenFn> =>
    listen<string>("meetings-answer-token", (e) => cb(e.payload)),
  meetingStatus: (
    cb: (s: { positionMs: number; pending: number; systemAudio: boolean }) => void,
  ): Promise<UnlistenFn> =>
    listen<{ positionMs: number; pending: number; systemAudio: boolean }>("meeting-status", (e) =>
      cb(e.payload),
    ),
  meetingWarning: (cb: (msg: string) => void): Promise<UnlistenFn> =>
    listen<string>("meeting-warning", (e) => cb(e.payload)),
  meetingSpeakerNamed: (cb: (p: { meetingId: string; speaker: string; name: string }) => void): Promise<UnlistenFn> =>
    listen<{ meetingId: string; speaker: string; name: string }>("meeting-speaker-named", (e) => cb(e.payload)),
  meetingNotesChanged: (cb: (p: { meetingId: string; notes: string }) => void): Promise<UnlistenFn> =>
    listen<{ meetingId: string; notes: string }>("meeting-notes-changed", (e) => cb(e.payload)),
  meetingNotesProgress: (
    cb: (p: { meetingId: string; throughMs: number }) => void,
  ): Promise<UnlistenFn> =>
    listen<{ meetingId: string; throughMs: number }>("meeting-notes-progress", (e) => cb(e.payload)),
  meetingDetected: (cb: (p: { app: string; bundleId: string }) => void): Promise<UnlistenFn> =>
    listen<{ app: string; bundleId: string }>("meeting-detected", (e) => cb(e.payload)),
  meetingStartRequest: (cb: (p: { app: string; bundleId: string }) => void): Promise<UnlistenFn> =>
    listen<{ app: string; bundleId: string }>("meeting-start-request", (e) => cb(e.payload)),
  meetingAppReleased: (cb: (p: { app: string; bundleId: string }) => void): Promise<UnlistenFn> =>
    listen<{ app: string; bundleId: string }>("meeting-app-released", (e) => cb(e.payload)),
  taskUpdated: (cb: (t: Task) => void): Promise<UnlistenFn> =>
    listen<Task>("task-updated", (e) => cb(e.payload)),
  assistantToast: (cb: (text: string) => void): Promise<UnlistenFn> =>
    listen<string>("assistant-toast", (e) => cb(e.payload)),
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
  dictationLive: (cb: (text: string) => void): Promise<UnlistenFn> =>
    listen<string>("dictation-live", (e) => cb(e.payload)),
};

/// A meeting position as people read it: "4:05", or "1:02:03" past an hour.
export function clock(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

/// "4:05" / "1:02:03" → ms (null if it isn't a time).
export function parseClock(t: string): number | null {
  const parts = t.trim().split(":").map((p) => Number(p));
  if (parts.some((p) => !Number.isFinite(p))) return null;
  if (parts.length === 2 && parts[1] < 60) return (parts[0] * 60 + parts[1]) * 1000;
  if (parts.length === 3 && parts[1] < 60 && parts[2] < 60)
    return (parts[0] * 3600 + parts[1] * 60 + parts[2]) * 1000;
  return null;
}

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
