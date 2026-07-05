import { useEffect, useRef, useState } from "react";
import { on } from "./api";

export const WAVE_BARS = 24;

/// Rolling history of mic levels, scaled to 0..1 per bar.
export function useLevelHistory(active: boolean): number[] {
  const [bars, setBars] = useState<number[]>(() => Array(WAVE_BARS).fill(0));
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    const sub = on.audioLevel((rms) => {
      if (!activeRef.current) return;
      const scaled = Math.min(1, Math.sqrt(rms / 0.25));
      setBars((prev) => [...prev.slice(1), scaled]);
    });
    return () => {
      sub.then((un) => un());
    };
  }, []);

  useEffect(() => {
    if (!active) setBars(Array(WAVE_BARS).fill(0));
  }, [active]);

  return bars;
}

export function Wave({ bars, className = "" }: { bars: number[]; className?: string }) {
  return (
    <div className={"wave " + className} aria-hidden>
      {bars.map((v, i) => (
        <span
          key={i}
          className="wave-bar"
          style={{ height: `${Math.max(8, v * 100)}%` }}
        />
      ))}
    </div>
  );
}
