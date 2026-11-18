import { PointerEvent, useEffect, useRef, useState } from "react";
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import { api, clock, MeetingRecordingStatus, on } from "./api";
import { Icon, Logo } from "./ui";

/// Window sizes (logical px). They include the transparent margin the drop
/// shadow draws into; meeting_indicator.rs creates the window at PILL size.
const WIDTH = 352;
const PILL_HEIGHT = 64;
const CARD_HEIGHT = 108;
/// Pointer travel before a press on the label turns into a window drag.
const DRAG_SLOP = 4;
const FLAGGED_MS = 1600;

/// Lives in its own window, so hiding the main app never hides the reminder.
export default function MeetingIndicator() {
  const [recording, setRecording] = useState<MeetingRecordingStatus | null>(null);
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [released, setReleased] = useState<string | null>(null);
  const [flagged, setFlagged] = useState(false);
  const recordingId = useRef<string | null>(null);
  const stoppingRef = useRef(false);
  const flagTimer = useRef<number | undefined>(undefined);
  const press = useRef<{ x: number; y: number; dragged: boolean } | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    root.classList.add("meeting-indicator-root", "floating-root");
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
      window.clearTimeout(flagTimer.current);
      subs.forEach((sub) => sub.then((unlisten) => unlisten()));
      root.classList.remove("meeting-indicator-root", "floating-root");
    };
  }, []);

  // The card needs more room than the pill. Grow the window only while it
  // shows, so the empty area below the pill never swallows clicks. Restore
  // the position afterwards so the top-left corner stays put on every OS.
  const showCard = !!recording && (!!released || !!error);
  useEffect(() => {
    const win = getCurrentWindow();
    let cancelled = false;
    (async () => {
      try {
        const pos = await win.outerPosition();
        if (cancelled) return;
        await win.setSize(new LogicalSize(WIDTH, showCard ? CARD_HEIGHT : PILL_HEIGHT));
        await win.setPosition(pos);
      } catch {
        // The card still renders; it just may be clipped at the bottom.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [showCard]);

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

  const flag = () => {
    const id = recordingId.current;
    if (!id) return;
    api
      .addMark(id, "")
      .then(() => {
        setFlagged(true);
        window.clearTimeout(flagTimer.current);
        flagTimer.current = window.setTimeout(() => setFlagged(false), FLAGGED_MS);
      })
      .catch(() => {});
  };

  const keep = () => {
    setReleased(null);
    setError(null);
  };

  // A press on the label opens the live view; moving past the slop drags the
  // window instead (native drag swallows the release, so no click follows).
  const onPointerDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    press.current = { x: e.clientX, y: e.clientY, dragged: false };
  };
  const onPointerMove = (e: PointerEvent) => {
    const p = press.current;
    if (!p || p.dragged || !(e.buttons & 1)) return;
    if (Math.hypot(e.clientX - p.x, e.clientY - p.y) < DRAG_SLOP) return;
    p.dragged = true;
    getCurrentWindow().startDragging().catch(() => {});
  };
  const onOpen = () => {
    const dragged = press.current?.dragged;
    press.current = null;
    if (dragged) return;
    api.showMainWindow("live").catch(() => {});
  };

  if (!recording) return null;
  const elapsed = <time className="fl-time">{clock(recording.elapsedMs)}</time>;
  const stopLabel = stopping ? "Stopping…" : "Stop";

  if (showCard) {
    return (
      <div className="fl-stack fl-stack-indicator">
        <div className="fl-surface fl-ended" role="group" aria-label="Recording">
          <div className="fl-ended-head" data-tauri-drag-region>
            <Logo size={18} />
            <span className={"fl-ended-title" + (error ? " danger" : "")}>
              {error ? "Couldn’t stop recording" : `${released} left the call`}
            </span>
            {elapsed}
          </div>
          <div className="fl-ended-body">
            <span className="fl-ended-ask" role={error ? "alert" : syncError ? "status" : undefined}>
              {error ? "Try again." : syncError ? "Reconnecting…" : "Stop recording?"}
            </span>
            <button className="btn btn-quiet fl-btn" onClick={keep}>
              Keep
            </button>
            <button className="btn btn-primary fl-btn" onClick={stop} disabled={stopping}>
              {stopLabel}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fl-stack fl-stack-indicator">
      <div className="fl-surface fl-pill fl-indicator">
        <span className="fl-indicator-grip" data-tauri-drag-region title="Drag to move">
          <Logo size={18} />
          <span className="rec-dot" />
        </span>
        <button
          className="fl-indicator-open"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onClick={onOpen}
          aria-label="Open the live recording"
        >
          <span className={"fl-indicator-label" + (flagged ? " flagged" : "")}>{flagged ? "Flagged" : "Recording"}</span>
          {syncError ? <span className="fl-indicator-sync">Reconnecting…</span> : elapsed}
        </button>
        <span className="visually-hidden" role="status">
          {flagged ? "Flagged" : ""}
        </span>
        <button className="fl-icon-btn" onClick={flag} aria-label="Flag this moment" title="Flag this moment">
          <Icon name={flagged ? "check" : "flag"} size={14} />
        </button>
        <button className="btn fl-btn fl-stop" onClick={stop} disabled={stopping}>
          <span className="fl-stop-glyph" aria-hidden="true" />
          {stopLabel}
        </button>
      </div>
    </div>
  );
}
