import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { createNotesDraft, type SaveState } from "./notesDraft";

export function useNotesSave(id: string, onRestore: (text: string) => void, onSaved?: (text: string) => void) {
  const [state, setState] = useState<SaveState>({ saving: false, error: null });
  const timer = useRef<number | undefined>(undefined);
  const mounted = useRef(true);
  const activeId = useRef(id);
  activeId.current = id;
  const saved = useRef(onSaved);
  saved.current = onSaved;
  const restore = useRef(onRestore);
  restore.current = onRestore;
  const draft = useMemo(() => createNotesDraft(id, {
    getItem: (key) => localStorage.getItem(key),
    setItem: (key, value) => localStorage.setItem(key, value),
    removeItem: (key) => localStorage.removeItem(key),
  }, api.updateMeetingNotes, (next) => {
    if (mounted.current && activeId.current === id) setState(next);
  }, (text) => { if (mounted.current && activeId.current === id) saved.current?.(text); }), [id]);

  const flush = () => { window.clearTimeout(timer.current); void draft.flush(); };
  useEffect(() => {
    mounted.current = true;
    setState({ saving: false, error: draft.initialError });
    if (draft.restored !== null) restore.current(draft.restored);
    return () => { mounted.current = false; window.clearTimeout(timer.current); void draft.flush(); };
  }, [draft]);

  const save = (text: string) => {
    draft.edit(text);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => { void draft.flush(); }, 500);
  };
  return { save, flush, ...state };
}
