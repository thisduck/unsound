import { useEffect, useState } from "react";
import { on, OverlayState } from "./api";
import { useLevelHistory, Wave } from "./Wave";
import "./App.css";

/// Rendered in the tiny always-on-top window while dictating into other apps.
export default function Overlay() {
  const [state, setState] = useState<OverlayState>("recording");
  const bars = useLevelHistory(state === "recording");

  useEffect(() => {
    document.documentElement.classList.add("overlay-root");
    const sub = on.overlayState(setState);
    return () => {
      sub.then((un) => un());
    };
  }, []);

  return (
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
  );
}
