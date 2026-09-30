import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { on, OverlayState } from "./api";
import { useLevelHistory, Wave } from "./Wave";
import "./App.css";

/// Rendered in the tiny always-on-top window while dictating into other apps:
/// the wave, with the words so far streaming in a bubble above it.
export default function Overlay() {
  const [state, setState] = useState<OverlayState>("recording");
  const [caption, setCaption] = useState("");
  const stateRef = useRef(state);
  stateRef.current = state;
  const bars = useLevelHistory(state === "recording");

  useEffect(() => {
    document.documentElement.classList.add("overlay-root");
    const subs = [
      on.overlayState((s) => {
        // A new take starts blank; the last caption stays up while it's processed.
        if (s !== "processing") setCaption("");
        setState(s);
      }),
      on.dictationLive((t) => {
        if (stateRef.current === "recording") setCaption(t);
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

  return (
    <div className="overlay-stack">
      {caption && (
        <div ref={captionRef} className={"overlay-caption" + (clipped ? " clipped" : "")}>
          <p>{caption}</p>
        </div>
      )}
      <div className="overlay-pill">
        {state === "processing" ? (
          <div className="overlay-processing">
            <span className="overlay-dot" />
            <span className="overlay-dot" />
            <span className="overlay-dot" />
          </div>
        ) : (
          <Wave bars={bars} className="overlay-wave" />
        )}
      </div>
    </div>
  );
}
