import type { Meeting } from "../api";

type People = Pick<Meeting, "participants" | "attendees" | "speakerNames">;

const key = (s: string) => s.trim().toLowerCase();

function unique(names: string[]): string[] {
  const seen = new Set<string>();
  return names.filter((n) => {
    const k = key(n);
    if (!k || seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/// Everyone but you seen on the call's participant list, in order of arrival.
export function onCall(m: People): string[] {
  return unique((m.participants?.people ?? []).filter((p) => !p.me).map((p) => p.name));
}

/// Who was in the meeting: people seen on the call, then calendar invitees.
export function meetingPeople(m: People): string[] {
  return unique([...onCall(m), ...(m.attendees ?? [])]);
}

/// Names to offer when naming a speaker: the call's roster, then invitees
/// (not email addresses), minus names already given to a speaker.
export function speakerSuggestions(m: People): string[] {
  const used = new Set(Object.values(m.speakerNames ?? {}).map(key));
  return meetingPeople(m).filter((n) => !n.includes("@") && !used.has(key(n)));
}

/// How many remote voices to expect, for re-detecting speakers: the people on
/// the call, else the calendar invitees. Null when unknown or out of range.
export function remoteCountHint(m: People): { count: number; from: "call" | "calendar" } | null {
  const call = onCall(m).length;
  const invited = unique(m.attendees ?? []).length;
  const hint = call > 0 ? { count: call, from: "call" as const } : invited > 0 ? { count: invited, from: "calendar" as const } : null;
  return hint && hint.count <= 8 ? hint : null;
}
