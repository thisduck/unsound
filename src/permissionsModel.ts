// What each macOS permission is for, and the pure decisions the onboarding
// step and Settings › Permissions make from its status. No React or IPC here.

export type PermissionKind =
  | "microphone"
  | "accessibility"
  | "screenRecording"
  | "calendar"
  | "automation"
  | "notifications";

export type Access = "granted" | "denied" | "notDetermined" | "restricted" | "unknown" | "unsupported";

export interface PermissionEntry {
  kind: PermissionKind;
  access: Access;
  /** Calendar only: whether "Use my calendar" is on. */
  enabled?: boolean;
}

export type PermissionIcon = "mic" | "keyboard" | "monitor" | "calendar" | "mail" | "bell";

export interface PermissionDef {
  kind: PermissionKind;
  title: string;
  why: string;
  icon: PermissionIcon;
  required: boolean;
  /** Shown during first-run setup (the rest live only in Settings). */
  onboarding: boolean;
  /** macOS applies a new grant only after the app reopens. */
  relaunch?: boolean;
  /** Label when macOS can't report a status. */
  unknownLabel?: string;
}

export const PERMISSIONS: PermissionDef[] = [
  {
    kind: "microphone",
    title: "Microphone",
    why: "Hears you when you dictate and records your side of a call.",
    icon: "mic",
    required: true,
    onboarding: true,
  },
  {
    kind: "accessibility",
    title: "Accessibility",
    why: "Types your words into the app you're using, runs fn-key shortcuts, and sees which call you're in and who's on it.",
    icon: "keyboard",
    required: true,
    onboarding: true,
  },
  {
    kind: "screenRecording",
    title: "Screen & System Audio Recording",
    why: "Hears the other people on a call. unsound records audio only, never your screen. Without it, meetings record your mic only.",
    icon: "monitor",
    required: false,
    onboarding: true,
    relaunch: true,
  },
  {
    kind: "calendar",
    title: "Calendars",
    why: "Names meetings and attendees from the event you're in.",
    icon: "calendar",
    required: false,
    onboarding: true,
  },
  {
    kind: "automation",
    title: "Automation",
    why: "Lets the assistant draft emails in Mail and add reminders to Reminders when you ask.",
    icon: "mail",
    required: false,
    onboarding: false,
    unknownLabel: "Asks on first use",
  },
  {
    kind: "notifications",
    title: "Notifications",
    why: "Reminds you about follow-ups while unsound is running.",
    icon: "bell",
    required: false,
    onboarding: false,
    unknownLabel: "Set in System Settings",
  },
];

export const defFor = (kind: PermissionKind) => PERMISSIONS.find((p) => p.kind === kind)!;

/** What the row's button does. */
export type PermissionAction = "allow" | "settings" | "enable" | "none";

// Accessibility and Screen Recording prompt once; after that only System
// Settings can change them, and macOS doesn't say whether the prompt was shown.
const PROMPTS_ONCE: PermissionKind[] = ["accessibility", "screenRecording", "automation"];

export function actionFor(entry: PermissionEntry | undefined, asked: boolean): PermissionAction {
  if (!entry) return "none";
  switch (entry.access) {
    case "granted":
      return entry.kind === "calendar" && entry.enabled === false ? "enable" : "none";
    case "unsupported":
      return "none";
    case "denied":
    case "restricted":
      return "settings";
    case "notDetermined":
      return asked && PROMPTS_ONCE.includes(entry.kind) ? "settings" : "allow";
    case "unknown":
      return entry.kind === "automation" && !asked ? "allow" : "settings";
  }
}

export type Tone = "ok" | "warn" | "danger" | "neutral";

export function statusLabel(def: PermissionDef, entry: PermissionEntry | undefined): { text: string; tone: Tone } | null {
  if (!entry) return null;
  switch (entry.access) {
    case "granted":
      return entry.kind === "calendar" && entry.enabled === false
        ? { text: "Allowed · not in use", tone: "neutral" }
        : { text: "Allowed", tone: "ok" };
    case "denied":
      return { text: "Off", tone: "warn" };
    case "restricted":
      return { text: "Blocked", tone: "danger" };
    case "notDetermined":
      return { text: "Not allowed", tone: "neutral" };
    case "unknown":
      return { text: def.unknownLabel ?? "Unknown", tone: "neutral" };
    case "unsupported":
      return { text: "Needs macOS 13", tone: "neutral" };
  }
}

/** The description line: why it's needed, or what to do when it's off. */
export function detailFor(def: PermissionDef, entry: PermissionEntry | undefined): string {
  if (entry?.access === "denied") return `Turned off. Switch unsound on in System Settings. ${def.why}`;
  if (entry?.access === "restricted") return "Your Mac's administrator controls this setting.";
  if (entry?.kind === "calendar" && entry.access === "granted" && entry.enabled === false)
    return "Allowed, but “Use my calendar” is off in Meetings settings.";
  return def.why;
}

const missing = (entry: PermissionEntry | undefined) =>
  !!entry && (entry.access === "notDetermined" || entry.access === "denied" || entry.access === "restricted");

/** Required permissions that are switched off right now. */
export function missingRequired(entries: PermissionEntry[]): PermissionKind[] {
  return PERMISSIONS.filter((d) => d.required && missing(entries.find((e) => e.kind === d.kind))).map((d) => d.kind);
}

/**
 * Existing users see the permissions check at launch only when a required
 * permission is off that they haven't already been shown. Fresh installs get
 * it inside onboarding instead.
 */
export function needsReview(onboarded: boolean, reviewed: PermissionKind[], entries: PermissionEntry[]): boolean {
  if (!onboarded) return false;
  return missingRequired(entries).some((k) => !reviewed.includes(k));
}

/** Show "relaunch unsound" once a relaunch-only permission was asked for and still reads as off. */
export function needsRelaunch(entries: PermissionEntry[], asked: PermissionKind[]): PermissionKind[] {
  return PERMISSIONS.filter((d) => {
    const e = entries.find((x) => x.kind === d.kind);
    return d.relaunch && asked.includes(d.kind) && !!e && e.access !== "granted" && e.access !== "unsupported";
  }).map((d) => d.kind);
}

export const OPEN_PERMISSIONS_EVENT = "unsound:open-permissions";
/** Open Settings › Permissions from anywhere in the app. */
export const openPermissions = () => window.dispatchEvent(new Event(OPEN_PERMISSIONS_EVENT));

export const ASKED_KEY = "unsound.permissions.asked";
export const REVIEWED_KEY = "unsound.permissions.reviewed";

interface Store {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function readKinds(store: Store | undefined, key: string): PermissionKind[] {
  try {
    const v = JSON.parse(store?.getItem(key) ?? "[]");
    return Array.isArray(v) ? v.filter((k) => PERMISSIONS.some((d) => d.kind === k)) : [];
  } catch {
    return [];
  }
}

export function addKinds(store: Store | undefined, key: string, kinds: PermissionKind[]): PermissionKind[] {
  const next = [...new Set([...readKinds(store, key), ...kinds])];
  try {
    store?.setItem(key, JSON.stringify(next));
  } catch {
    /* storage unavailable; this session still remembers */
  }
  return next;
}
