import { useEffect, useRef, useState } from "react";
import { api, clock, ResearchAnswer, ResearchSource } from "../api";
import { Icon, Spinner } from "../ui";
import { speakerLabel } from "./format";

interface Props {
  modelId?: string;
  /// Ask about one meeting; otherwise the whole archive.
  meetingId?: string;
  initialQuestion?: string;
  /// Ask initialQuestion as soon as the panel opens.
  autoAsk?: boolean;
  onSource: (id: string, at?: number | null) => void;
  onChooseModel?: () => void;
}

/// Questions answered from transcripts and summaries, with every claim
/// linked back to its source.
export function ResearchPanel({ modelId, meetingId, initialQuestion = "", autoAsk, onSource, onChooseModel }: Props) {
  const [question, setQuestion] = useState(autoAsk ? "" : initialQuestion);
  const [turns, setTurns] = useState<{ question: string; answer: ResearchAnswer }[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inFlight = useRef(false);

  const ask = async (text = question) => {
    const q = text.trim();
    if (!modelId || inFlight.current || !q) return;
    inFlight.current = true;
    setBusy(true);
    setError("");
    try {
      const history = turns.slice(-4).map((t) => ({ question: t.question, answer: [...t.answer.claims.map((c) => c.text), ...t.answer.quotes.map((x) => x.text)].join("\n") }));
      const answer = await api.researchMeetings(modelId, q, meetingId ?? null, history);
      setTurns((t) => [...t, { question: q, answer }]);
      setQuestion("");
    } catch (e) {
      setError(String(e));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  };

  useEffect(() => {
    if (autoAsk && initialQuestion.trim()) void ask(initialQuestion);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const label = (s: ResearchSource) =>
    `${s.title || "Untitled meeting"} · ${new Date(s.date).toLocaleDateString()}${s.atMs != null ? ` · ${clock(s.atMs)}` : " · summary"}`;

  return (
    <div className="research">
      {!modelId && (
        <div className="alert" role="status">
          <Icon name="info" />
          <span className="alert-body">Answering questions needs a summary model.</span>
          {onChooseModel && (
            <div className="alert-actions">
              <button className="btn btn-secondary btn-sm" onClick={onChooseModel}>Choose a model</button>
            </div>
          )}
        </div>
      )}
      {turns.map((turn, n) => (
        <div className="research-turn" key={n}>
          <h4>{turn.question}</h4>
          {turn.answer.claims.map((claim, i) => (
            <p key={i}>
              {claim.text}{" "}
              {claim.sourceIds.map((id) => {
                const s = turn.answer.sources.find((s) => s.id === id);
                return s && (
                  <button className="time-link" title={label(s)} key={id} onClick={() => onSource(s.meetingId, s.atMs)}>
                    {turn.answer.sources.indexOf(s) + 1}
                  </button>
                );
              })}
            </p>
          ))}
          {turn.answer.quotes.map((q, i) => {
            const s = turn.answer.sources.find((s) => s.id === q.sourceId);
            return s && (
              <figure className="research-quote" key={i}>
                <blockquote>{q.text}</blockquote>
                <figcaption>
                  <button className="btn btn-quiet btn-sm" onClick={() => onSource(s.meetingId, s.atMs)}>
                    {s.speaker ? `${/^(me|them(?::\d+)?|manual:.+)$/.test(s.speaker) ? speakerLabel(s.speaker) : s.speaker} · ` : ""}
                    {label(s)}
                  </button>
                </figcaption>
              </figure>
            );
          })}
          {turn.answer.insufficient && (
            <p className="faint">The sources don't fully answer this. Try a person, topic or meeting name.</p>
          )}
          {!!turn.answer.sources.length && (
            <details className="research-sources">
              <summary>{turn.answer.sources.length} source{turn.answer.sources.length === 1 ? "" : "s"}</summary>
              {turn.answer.sources.map((s, i) => (
                <div className="research-source" key={s.id}>
                  <button className="btn btn-quiet btn-sm" onClick={() => onSource(s.meetingId, s.atMs)}>
                    {i + 1}. {label(s)}
                  </button>
                  {s.kind === "summary" && <span className="faint">{s.stale ? "Summary is out of date" : "Summary"}</span>}
                  <p>{s.text}</p>
                </div>
              ))}
            </details>
          )}
        </div>
      ))}
      {busy && (
        <p className="research-busy" role="status">
          <Spinner /> Searching your meetings and checking sources…
        </p>
      )}
      {error && (
        <div className="alert danger" role="alert">
          <Icon name="alert" />
          <span className="alert-body">{error}</span>
        </div>
      )}
      {(!autoAsk || turns.length > 0 || error) && (
        <form className="research-ask" onSubmit={(e) => { e.preventDefault(); void ask(); }}>
          <input
            className="input"
            aria-label={meetingId ? "Ask about this meeting" : "Ask your meetings"}
            value={question}
            maxLength={4000}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={turns.length ? "Ask a follow-up…" : meetingId ? "What did they decide about pricing?" : "What did we decide about pricing?"}
            disabled={busy || !modelId}
          />
          <button className="btn btn-secondary" disabled={busy || !modelId || !question.trim()}>Ask</button>
          {turns.length > 0 && (
            <button type="button" className="btn btn-quiet" disabled={busy} onClick={() => setTurns([])}>New conversation</button>
          )}
        </form>
      )}
    </div>
  );
}
