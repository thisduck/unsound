import { useEffect, useRef, useState } from "react";
import { api, clock, ResearchAnswer, ResearchSource } from "../api";
import { speakerLabel } from "./format";

export function ResearchPanel({ modelId, meetingId, initialQuestion = "", onSource }: { modelId: string; meetingId?: string; initialQuestion?: string; onSource: (id: string, at?: number | null) => void }) {
  const [question, setQuestion] = useState(initialQuestion);
  const [turns, setTurns] = useState<{ question: string; answer: ResearchAnswer }[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inFlight = useRef(false);
  useEffect(() => { if (!turns.length && !inFlight.current) setQuestion(initialQuestion); }, [initialQuestion]);
  const ask = async () => {
    if (inFlight.current || !question.trim()) return;
    const q = question.trim(); inFlight.current = true; setBusy(true); setError("");
    try {
      const history = turns.slice(-4).map(t => ({ question: t.question, answer: [...t.answer.claims.map(c => c.text), ...t.answer.quotes.map(x => x.text)].join("\n") }));
      const answer = await api.researchMeetings(modelId, q, meetingId ?? null, history);
      setTurns(t => [...t, { question: q, answer }]); setQuestion("");
    } catch (e) { setError(String(e)); }
    finally { inFlight.current = false; setBusy(false); }
  };
  const label = (s: ResearchSource) => `${s.title || "Untitled meeting"} · ${new Date(s.date).toLocaleDateString()}${s.atMs != null ? ` · ${clock(s.atMs)}` : " · summary"}`;
  return <section className="meet-section research-panel">
    <div className="meet-section-head"><h3>{meetingId ? "ask this meeting" : "ask your meetings"}</h3>{turns.length > 0 && <button className="quiet" disabled={busy} onClick={() => setTurns([])}>new conversation</button>}</div>
    <p className="meet-hint">Search transcripts and summaries, compare decisions, or pull exact quotes. Answers link back to the evidence.</p>
    {turns.map((turn, n) => <div className="research-turn" key={n}>
      <h4>{turn.question}</h4>
      {turn.answer.claims.map((claim, i) => <p key={i}>{claim.text} {claim.sourceIds.map(id => { const s = turn.answer.sources.find(s => s.id === id); return s && <button className="time-link" title={label(s)} key={id} onClick={() => onSource(s.meetingId, s.atMs)}>[{turn.answer.sources.indexOf(s) + 1}]</button>; })}</p>)}
      {turn.answer.quotes.map((q, i) => { const s = turn.answer.sources.find(s => s.id === q.sourceId); return s && <figure className="research-quote" key={i}><blockquote>{q.text}</blockquote><figcaption><button className="quiet" onClick={() => onSource(s.meetingId, s.atMs)}>{s.speaker ? `${/^(me|them(?::\d+)?|manual:.+)$/.test(s.speaker) ? speakerLabel(s.speaker) : s.speaker} · ` : ""}{label(s)}</button></figcaption></figure>; })}
      {turn.answer.insufficient && <p className="meet-hint" role="status">The retrieved sources do not fully answer this question. Try a person, topic, or meeting name to narrow the search.</p>}
      {!!turn.answer.sources.length && <details><summary>{turn.answer.sources.length} sources · {turn.answer.searches.length} searches / reads</summary>{turn.answer.sources.map((s, i) => <div className="research-source" key={s.id}><button className="quiet" onClick={() => onSource(s.meetingId, s.atMs)}>[{i + 1}] {label(s)}</button>{s.kind === "summary" && <span className="meet-hint">{s.stale ? "Summary needs regeneration" : "Generated summary"}</span>}<p>{s.text}</p></div>)}</details>}
    </div>)}
    <form className="meet-ask-row" onSubmit={e => { e.preventDefault(); void ask(); }}>
      <input className="meet-ask-input" aria-label="Ask AI about meetings" value={question} maxLength={4000} onChange={e => setQuestion(e.target.value)} placeholder={turns.length ? "Ask a follow-up…" : "What did we decide about pricing? Pull the exact quotes."} disabled={busy} />
      <button className="quiet accent" disabled={busy || !question.trim()}>{busy ? "researching…" : "ask"}</button>
    </form>
    {busy && <p className="dim" role="status">Searching meetings and checking sources…</p>}
    {error && <p className="meet-error" role="alert">{error}</p>}
  </section>;
}
