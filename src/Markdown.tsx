import { Fragment, ReactNode } from "react";
import { parseClock } from "./api";

/// A deliberately small Markdown renderer for model-written notes: headings,
/// bullet/numbered lists (one level of nesting), checkboxes, **bold**,
/// *italic*, `code`, paragraphs. Builds React elements — never raw HTML — so
/// model output can't inject markup. Transcript times in parentheses or
/// brackets — "(12:34)", "[0:25, 0:37]" — become links when `onTime` is set.
export function Markdown({
  text,
  onTime,
  className,
}: {
  text: string;
  onTime?: (ms: number) => void;
  className?: string;
}) {
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: ReactNode[] } | null = null;
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      blocks.push(<p key={blocks.length}>{inline(para.join(" "), onTime)}</p>);
      para = [];
    }
  };
  const flushList = () => {
    if (list) {
      const items = list.items.map((it, i) => <li key={i}>{it}</li>);
      blocks.push(list.ordered ? <ol key={blocks.length}>{items}</ol> : <ul key={blocks.length}>{items}</ul>);
      list = null;
    }
  };

  for (const raw of text.split("\n")) {
    const line = raw.replace(/\s+$/, "");
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const bullet = line.match(/^(\s*)[-*•]\s+(.*)$/);
    const numbered = line.match(/^(\s*)\d+[.)]\s+(.*)$/);
    if (!line.trim()) {
      flushPara();
      flushList();
    } else if (heading) {
      flushPara();
      flushList();
      blocks.push(<h4 key={blocks.length}>{inline(heading[1], onTime)}</h4>);
    } else if (bullet || numbered) {
      flushPara();
      const m = (bullet ?? numbered)!;
      const ordered = !bullet;
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      const nested = m[1].length >= 2;
      const body = m[2];
      const box = body.match(/^\[( |x|X)\]\s+(.*)$/);
      const content = box ? (
        <span className="md-check">
          <span className={"md-box" + (box[1] !== " " ? " on" : "")} />
          {inline(box[2], onTime)}
        </span>
      ) : (
        inline(body, onTime)
      );
      list.items.push(nested ? <span className="md-nested">{content}</span> : content);
    } else {
      flushList();
      para.push(line.trim());
    }
  }
  flushPara();
  flushList();
  return <div className={"md " + (className ?? "")}>{blocks}</div>;
}

const TOKEN = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*|_[^_\s][^_]*_|[([](?:\d{1,2}:)?\d{1,2}:\d{2}(?:\s*[,–-]\s*(?:\d{1,2}:)?\d{1,2}:\d{2})*[)\]])/g;

function inline(text: string, onTime?: (ms: number) => void): ReactNode {
  const out: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    const tok = m[0];
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    if (tok.startsWith("**")) out.push(<strong key={at}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) out.push(<code key={at}>{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("*") || tok.startsWith("_")) out.push(<em key={at}>{tok.slice(1, -1)}</em>);
    else out.push(<Times key={at} raw={tok} onTime={onTime} />);
    last = at + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function Times({ raw, onTime }: { raw: string; onTime?: (ms: number) => void }) {
  const inner = raw.slice(1, -1);
  if (!onTime) return <span className="md-time">{raw}</span>;
  const parts = inner.split(/(\s*[,–-]\s*)/);
  return (
    <span className="md-time">
      {raw[0]}
      {parts.map((p, i) => {
        const ms = parseClock(p);
        return ms === null ? (
          <Fragment key={i}>{p}</Fragment>
        ) : (
          <button key={i} className="time-link" onClick={() => onTime(ms)} title="Play from here">
            {p}
          </button>
        );
      })}
      {raw[raw.length - 1]}
    </span>
  );
}
