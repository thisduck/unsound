// Shared across live/archive editors. Requests for one meeting run in order.
const writes = new Map<string, Promise<void>>();
const revisions = new Map<string, number>();
export type SaveState = { saving: boolean; error: string | null };
type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function createNotesDraft(
  id: string, storage: DraftStorage, write: (id: string, text: string) => Promise<void>,
  report: (state: SaveState) => void,
  saved?: (text: string) => void,
) {
  const key = `unsound.notes-draft.${id}`;
  let pending: string | null = null;
  let revision = 0;
  let initialError: string | null = null;
  const advance = () => {
    revision = (revisions.get(id) ?? 0) + 1;
    revisions.set(id, revision);
  };
  try {
    pending = storage.getItem(key);
    if (pending !== null) { advance(); initialError = "Recovered an unsaved notes draft. Review it, then retry saving."; }
  } catch { initialError = "Local draft storage is unavailable. Keep this window open until your notes finish saving."; }
  const restored = pending;

  return {
    restored, initialError,
    edit(text: string) {
      pending = text;
      advance();
      let error = null;
      try { storage.setItem(key, text); }
      catch { error = "The local draft could not be saved. Keep this window open until the meeting save succeeds."; }
      report({ saving: true, error });
    },
    flush(): Promise<void> {
      if (pending === null) return Promise.resolve();
      const text = pending;
      const sent = revision;
      const task = (writes.get(id) ?? Promise.resolve()).then(async () => {
        try {
          await write(id, text);
          if (revision === sent && revisions.get(id) === sent) {
            pending = null;
            try { if (storage.getItem(key) === text) storage.removeItem(key); } catch { /* database save succeeded */ }
            saved?.(text);
            report({ saving: false, error: null });
          }
        } catch (e) {
          report({ saving: false, error: `Notes are not saved to the meeting: ${String(e)}. Retry saving; keep this window open if local draft storage is unavailable.` });
        }
      });
      writes.set(id, task);
      void task.finally(() => { if (writes.get(id) === task) writes.delete(id); });
      return task;
    },
  };
}
