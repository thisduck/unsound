import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { api, on, OverlayState } from "./api";
import { useLevelHistory, Wave } from "./Wave";
import "./App.css";

/// Rendered in the tiny always-on-top window while dictating into other apps:
/// the wave, with the words so far streaming in a bubble above it.
export default function Overlay() {
  const [state, setState] = useState<OverlayState>("hidden");
  const [caption, setCaption] = useState("");
  // Assistant acknowledgements ("Got it: …", "Created ENG-142").
  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<number | undefined>(undefined);
  const stateRef = useRef(state);
  const takeRef = useRef<string | undefined>(undefined);
  stateRef.current = state;
  const bars = useLevelHistory(state === "recording" || state === "command");

  useEffect(() => {
    document.documentElement.classList.add("overlay-root");
    const subs = [
      on.overlayState((s, takeId) => {
        // A new take starts blank; the last caption stays up while it's processed.
        if (s !== "processing") setCaption("");
        takeRef.current = takeId;
        stateRef.current = s;
        setState(s);
      }),
      on.dictationLive(({ takeId, text }) => {
        if (stateRef.current === "recording" && takeRef.current === takeId) setCaption(text);
      }),
      on.assistantToast((text) => {
        setToast(text);
        window.clearTimeout(toastTimer.current);
        toastTimer.current = window.setTimeout(() => {
          setToast(null);
          // Nothing else on screen: put the overlay away again.
          if (stateRef.current === "hidden" || stateRef.current === "processing") {
            setState("hidden");
            api.setOverlay(false).catch(() => {});
          }
        }, 4000);
      }),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
  }, []);

  // Only the latest lines fit; fade the top edge once earlier ones are cut off.
  const captionRef = useRef<HTMLDivElement>(null);
  const [clipped, setClipped] = useState(false);
  useLayoutEffect(() => {
    // Text cut off at the top doesn't count toward scrollHeight, so compare
    // the paragraph's height with the room inside the padding instead.
    const el = captionRef.current;
    const text = el?.firstElementChild as HTMLElement | null;
    if (!el || !text) return setClipped(false);
    const style = getComputedStyle(el);
    const room = el.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    setClipped(text.offsetHeight > room + 1);
  }, [caption]);

  const showPill = state === "recording" || state === "processing" || state === "command";
  return (
    <div className="overlay-stack">
      {toast && (
        <div className="overlay-caption overlay-toast">
          <p>{toast}</p>
        </div>
      )}
      {!toast && caption && (
        <div ref={captionRef} className={"overlay-caption" + (clipped ? " clipped" : "")}>
          <p>{caption}</p>
        </div>
      )}
      {showPill && <div className={"overlay-pill" + (state === "command" ? " command" : "")}>
        {state === "command" && <span className="overlay-label">ask</span>}
        {state === "processing" ? (
          <div className="overlay-processing">
            <span className="overlay-dot" />
            <span className="overlay-dot" />
            <span className="overlay-dot" />
          </div>
        ) : (
          <Wave bars={bars} className="overlay-wave" />
        )}
      </div>}
    </div>
  );
}
