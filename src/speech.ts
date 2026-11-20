import type { ModelInfo } from "./api";
import { LANGUAGES } from "./languages.ts";

/// Removed speech model ids and their replacements (mirrors `models::RENAMED`).
/// `parakeet-tdt-v3` was the sherpa-onnx build of Parakeet v3.
const RENAMED: Record<string, string> = { "parakeet-tdt-v3": "parakeet-tdt-0.6b-v3" };

export function migrateSttId(id: string): string {
  return RENAMED[id] ?? id;
}

/// Whisper models that stand in for a model without the spoken language,
/// most accurate first (mirrors `models::FALLBACK_WHISPER`).
const FALLBACK_WHISPER = [
  "whisper-large-v3-turbo",
  "whisper-large-v3",
  "whisper-medium",
  "whisper-small",
  "whisper-base",
  "whisper-tiny",
];

export function fallbackWhisper(models: ModelInfo[]): ModelInfo | undefined {
  for (const id of FALLBACK_WHISPER) {
    const m = models.find((x) => x.id === id && x.downloaded);
    if (m) return m;
  }
  return undefined;
}

export function languageName(code: string): string {
  return LANGUAGES.find((l) => l.code === code && code !== "")?.name ?? code;
}

/// The languages a model covers, by name, for its card ("" when it takes any).
export function supportedLanguageNames(model: ModelInfo): string {
  const codes = model.supportedLanguages ?? [];
  return codes.map(languageName).sort().join(", ");
}

/// What happens to speech in `code` ("" = auto-detect) with this speech model,
/// when that is not simply "the model transcribes it". Mirrors `asr::route`.
export function languageNote(
  model: ModelInfo | undefined,
  code: string,
  models: ModelInfo[],
): { tone: "info" | "warn"; text: string } | null {
  const supported = model?.supportedLanguages ?? [];
  if (!model || supported.length === 0) return null;
  const fallback = fallbackWhisper(models);
  const assumed = supported.length === 1 ? languageName(supported[0]) : `one of its ${supported.length} languages`;
  if (code === "") {
    if (fallback) return null;
    return {
      tone: "info",
      text: `Auto-detect needs a Whisper model. Without one, ${model.name} treats all speech as ${assumed}.`,
    };
  }
  if (supported.includes(code)) return null;
  if (fallback) {
    return { tone: "info", text: `${model.name} doesn't do ${languageName(code)}, so ${fallback.name} transcribes it.` };
  }
  return {
    tone: "warn",
    text: `${model.name} can't transcribe ${languageName(code)}. Download a Whisper model in Settings › Models, or pick another speech model.`,
  };
}
