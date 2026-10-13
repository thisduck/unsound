import { clock, formatDuration, Meeting } from "../api";

/// How a segment's speaker is shown. A per-meeting name wins; otherwise "You"
/// for the mic, "Speaker N" for a detected participant (`them:0` → Speaker 1),
/// else "Them".
export function speakerLabel(speaker: string, names?: Record<string, string>): string {
  const custom = names?.[speaker];
  if (custom) return custom;
  if (speaker === "me") return "You";
  const m = speaker.match(/^them:(\d+)$/);
  if (m) return `Speaker ${Number(m[1]) + 1}`;
  return "Them";
}

/// Stable-ish color class per speaker so the transcript is easy to scan.
export function speakerClass(speaker: string): string {
  if (speaker === "me") return "seg-me";
  const m = speaker.match(/^them:(\d+)$/);
  if (m) return `seg-spk${Number(m[1]) % 6}`;
  return "seg-them";
}

export function elapsed(startedAt: string, endedAt?: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = endedAt ? new Date(endedAt).getTime() : Date.now();
  return formatDuration(Math.max(0, (end - start) / 1000));
}

/// The whole meeting as shareable Markdown.
export function meetingMarkdown(m: Meeting): string {
  const out: string[] = [`# ${m.title || "Untitled meeting"}`, ""];
  out.push(`_${new Date(m.startedAt).toLocaleString()} · ${elapsed(m.startedAt, m.endedAt)}_`, "");
  if (m.attendees?.length) out.push(`**Attendees:** ${m.attendees.join(", ")}`, "");
  if (m.summary.trim()) out.push(m.summary.trim(), "");
  const items = m.actionItems ?? [];
  if (items.length) {
    out.push("## Action items", "");
    for (const it of items) {
      const meta = [it.owner, it.due].filter(Boolean).join(", ");
      out.push(`- [${it.done ? "x" : " "}] ${it.text}${meta ? ` (${meta})` : ""}`);
    }
    out.push("");
  }
  if (m.notes.trim()) out.push("## My notes", "", m.notes.trim(), "");
  if (m.segments.length) {
    out.push("## Transcript", "");
    for (const s of [...m.segments].sort((a, b) => a.startMs - b.startMs)) {
      out.push(`**${speakerLabel(s.speaker, m.speakerNames)}** (${clock(s.startMs)}): ${s.text}`);
    }
  }
  return out.join("\n");
}

/// Split a « »-marked FTS snippet into plain and highlighted runs.
export function snippetParts(snippet: string): { text: string; hit: boolean }[] {
  const out: { text: string; hit: boolean }[] = [];
  const re = /«([^»]*)»/g;
  let last = 0;
  for (const m of snippet.matchAll(re)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ text: snippet.slice(last, at), hit: false });
    out.push({ text: m[1], hit: true });
    last = at + m[0].length;
  }
  if (last < snippet.length) out.push({ text: snippet.slice(last), hit: false });
  return out;
}
