import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

/// How much audio each request fetches; playback chains the next piece.
const PIECE_MS = 90_000;

export interface Player {
  /// Where playback is on the meeting clock (null when stopped).
  atMs: number | null;
  playing: boolean;
  loading: boolean;
  error: string | null;
  play: (fromMs: number) => void;
  toggle: () => void;
  stop: () => void;
}

/// Play a meeting's recording from any moment. Audio comes from the backend in
/// ~90 s WAV pieces (both channels mixed), fetched as they're needed, so an
/// hour-long meeting never loads whole.
export function usePlayer(meetingId: string | null): Player {
  const audio = useRef<HTMLAudioElement | null>(null);
  const url = useRef<string | null>(null);
  const pieceStart = useRef(0);
  const lastRequest = useRef(0);
  const [atMs, setAtMs] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const release = () => {
    if (url.current) URL.revokeObjectURL(url.current);
    url.current = null;
  };

  const stop = useCallback(() => {
    audio.current?.pause();
    release();
    setPlaying(false);
    setAtMs(null);
  }, []);

  const play = useCallback(
    async (fromMs: number) => {
      if (!meetingId) return;
      const request = ++lastRequest.current;
      setLoading(true);
      setError(null);
      try {
        const start = Math.max(0, fromMs - 500);
        const buf = await api.meetingClip(meetingId, start, start + PIECE_MS);
        if (request !== lastRequest.current) return; // superseded by a newer click
        if (!audio.current) {
          audio.current = new Audio();
          audio.current.addEventListener("timeupdate", () => {
            const a = audio.current!;
            setAtMs(pieceStart.current + a.currentTime * 1000);
          });
          audio.current.addEventListener("play", () => setPlaying(true));
          audio.current.addEventListener("pause", () => setPlaying(false));
          audio.current.addEventListener("ended", () => {
            const a = audio.current!;
            // A full piece means there's more after it.
            if (a.duration * 1000 >= PIECE_MS - 1000) play(pieceStart.current + a.duration * 1000);
            else setPlaying(false);
          });
        }
        release();
        url.current = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
        pieceStart.current = start;
        audio.current.src = url.current;
        await audio.current.play();
      } catch (e) {
        if (request === lastRequest.current) setError(String(e));
      } finally {
        if (request === lastRequest.current) setLoading(false);
      }
    },
    [meetingId],
  );

  const toggle = useCallback(() => {
    const a = audio.current;
    if (!a || !url.current) return;
    if (a.paused) a.play();
    else a.pause();
  }, []);

  // A different meeting (or leaving it) stops playback.
  useEffect(() => stop, [meetingId, stop]);

  return { atMs, playing, loading, error, play, toggle, stop };
}
