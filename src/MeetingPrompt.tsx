import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import { Logo } from "./ui";

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
    document.documentElement.classList.add("prompt-root", "floating-root");
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
    <div className="fl-stack fl-stack-prompt">
      <div className="fl-surface fl-prompt" role="dialog" aria-labelledby="fl-prompt-title">
        <div className="fl-prompt-head">
          <Logo size={30} />
          <div className="fl-prompt-text">
            <div id="fl-prompt-title" className="fl-prompt-title">
              {detected ? `${detected.app} call started` : "A call may have started"}
            </div>
            <div className="fl-prompt-sub">Record it and get notes when it ends?</div>
          </div>
        </div>
        <div className="fl-prompt-actions">
          <button className="btn btn-quiet fl-btn" onClick={hide}>
            Not now
          </button>
          <button className="btn btn-primary fl-btn" onClick={record}>
            Record
          </button>
        </div>
      </div>
    </div>
  );
}
