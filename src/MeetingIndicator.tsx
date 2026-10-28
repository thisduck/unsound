import { useEffect, useRef, useState } from "react";
import { api, clock, MeetingRecordingStatus, on } from "./api";
import "./App.css";

/// Lives in its own window, so hiding the main app never hides the reminder.
export default function MeetingIndicator() {
  const [recording, setRecording] = useState<MeetingRecordingStatus | null>(null);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [released, setReleased] = useState<string | null>(null);
  const recordingId = useRef<string | null>(null);
  const stoppingRef = useRef(false);

  useEffect(() => {
    document.documentElement.classList.add("meeting-indicator-root");
    let mounted = true;
    let polling = false;
    const refresh = async () => {
      if (polling) return;
      polling = true;
      try {
        const current = await api.meetingRecordingStatus();
        if (!mounted) return;
        setSyncError(null);
        if (recordingId.current !== (current?.meetingId ?? null)) {
          setReleased(null);
          setError(null);
          stoppingRef.current = false;
          setStopping(false);
        }
        recordingId.current = current?.meetingId ?? null;
        setRecording(current);
      } catch (e) {
        // Keep the last known recording visible if a status read fails.
        if (mounted) setSyncError(String(e));
      } finally {
        polling = false;
      }
    };
    const subs = [
      on.meetingRecordingChanged(refresh),
      on.meetingAppReleased(({ app }) => {
        if (recordingId.current) setReleased(app);
      }),
      on.meetingDetected(() => setReleased(null)),
    ];
    void refresh();
    // Also recovers state after a webview reload or a missed start/stop event.
    const timer = window.setInterval(refresh, 1000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      subs.forEach((sub) => sub.then((unlisten) => unlisten()));
      document.documentElement.classList.remove("meeting-indicator-root");
    };
  }, []);

  const stop = async () => {
    const id = recordingId.current;
    if (!id || stoppingRef.current) return;
    stoppingRef.current = true;
    setStopping(true);
    setError(null);
    try {
      // Stop capture directly, even if the main webview is hidden or reloading.
      await api.meetingStop(id);
    } catch (e) {
      if (recordingId.current === id) setError(String(e));
    } finally {
      if (recordingId.current === id) {
        stoppingRef.current = false;
        setStopping(false);
      }
    }
  };

  if (!recording) return null;

  return (
    <div className="meeting-indicator-stack">
      <div className={"meeting-indicator" + (released ? " call-ended" : "")}>
        <div className="meeting-indicator-drag" data-tauri-drag-region title={error ?? syncError ?? (released ? `${released} released the mic. Recording is still running. Drag to move.` : "Drag to move recording indicator")}>
          <span className="meeting-indicator-brand" aria-hidden="true">
            <img src="/unsound.svg" alt="" />
            <span className="meeting-indicator-dot" />
          </span>
          <div className="meeting-indicator-copy">
            <strong>unsound is recording</strong>
            <span className="meeting-indicator-detail">
              {error ? <span role="alert">Couldn’t stop. Try again.</span> : syncError ? "Reconnecting…" : <>{released && <span role="status">Call ended? · </span>}<time>{clock(recording.elapsedMs)}</time></>}
            </span>
          </div>
        </div>
        <button className="meeting-indicator-stop" onClick={stop} disabled={stopping} aria-label="Stop meeting recording">
          <span aria-hidden="true" />{stopping ? "Stopping…" : "Stop"}
        </button>
      </div>
    </div>
  );
}
