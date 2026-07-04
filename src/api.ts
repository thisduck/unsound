import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

export type ModelKind = "stt" | "llm";

export interface ModelInfo {
  id: string;
  name: string;
  kind: ModelKind;
  url: string;
  filename: string;
  sizeBytes: number;
  description: string;
  custom: boolean;
  downloaded: boolean;
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

export interface Settings {
  shortcut: string;
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
  cleanupText: (modelId: string, text: string, prompt?: string) =>
    invoke<string>("cleanup_text", { modelId, text, prompt: prompt ?? null }),
  defaultCleanupPrompt: () => invoke<string>("default_cleanup_prompt"),
  getSettings: () => invoke<Settings>("get_settings"),
  setShortcut: (shortcut: string) => invoke<void>("set_shortcut", { shortcut }),
  deliverText: (text: string) => invoke<void>("deliver_text", { text }),
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
        case "space":
          return "Space";
        default:
          return part.length === 1 ? part.toUpperCase() : part;
      }
    })
    .join("");
}

export function formatBytes(n: number): string {
  if (!n) return "?";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}
