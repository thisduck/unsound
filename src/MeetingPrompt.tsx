import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import "./App.css";

interface Detected {
  app: string;
  bundleId: string;
}

/// Seconds before an unanswered prompt tucks itself away.
const AUTO_HIDE_MS = 20_000;

/// Rendered in the small top-right window shown when a call app takes the
/// mic: offer to record the meeting, and get out of the way otherwise.
export default function MeetingPrompt() {
  const [detected, setDetected] = useState<Detected | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const hide = () => {
    window.clearTimeout(timer.current);
    invoke("hide_meeting_prompt").catch(() => {});
  };

  useEffect(() => {
    document.documentElement.classList.add("prompt-root");
    const sub = listen<Detected>("meeting-detected", (e) => {
      setDetected(e.payload);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(hide, AUTO_HIDE_MS);
    });
    return () => {
      sub.then((un) => un());
      window.clearTimeout(timer.current);
    };
  }, []);

  const record = () => {
    if (detected) emit("meeting-start-request", detected).catch(() => {});
    hide();
  };

  return (
    <div className="prompt-stack">
      <div className="prompt-card">
        <div className="prompt-text">
          <div className="prompt-title">{detected ? `${detected.app} is using your mic` : "A call started"}</div>
          <div className="prompt-sub">Record this meeting? It stays on this Mac.</div>
        </div>
        <div className="prompt-actions">
          <button className="quiet" onClick={hide}>
            Not now
          </button>
          <button className="quiet accent" onClick={record}>
            Record
          </button>
        </div>
      </div>
    </div>
  );
}
