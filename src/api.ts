import { invoke } from "@tauri-apps/api/core";
import { emit, listen, UnlistenFn } from "@tauri-apps/api/event";

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

export const api = {
  listModels: () => invoke<ModelInfo[]>("list_models"),
  downloadModel: (id: string) => invoke<void>("download_model", { id }),
  deleteModel: (id: string) => invoke<void>("delete_model", { id }),
  addCustomModel: (name: string, kind: ModelKind, url: string) =>
    invoke<ModelInfo>("add_custom_model", { name, kind, url }),
  startRecording: () => invoke<void>("start_recording"),
  stopRecording: () => invoke<RecordingResult>("stop_recording"),
  transcribe: (modelId: string, language?: string) =>
    invoke<string>("transcribe", { modelId, language: language ?? null }),
  cleanupText: (modelId: string, text: string, prompt?: string, styleId?: string) =>
    invoke<string>("cleanup_text", {
      modelId,
      text,
      prompt: prompt ?? null,
      styleId: styleId ?? null,
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
};

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

export function formatBytes(n: number): string {
  if (!n) return "?";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}
