export interface AppUpdate { version: string; currentVersion: string; notes: string }
export type UpdatePhase = "idle" | "checking" | "current" | "available" | "downloading" | "ready" | "installing";
export interface UpdateState {
  phase: UpdatePhase; update: AppUpdate | null; error: string | null;
  downloaded: number; total: number | null;
}
export function createUpdateController(deps: {
  check: () => Promise<AppUpdate | null>;
  download: () => Promise<void>;
  install: () => Promise<void>;
  blocker: () => string | null;
  changed: (state: UpdateState) => void;
}) {
  let state: UpdateState = { phase: "idle", update: null, error: null, downloaded: 0, total: null };
  const set = (next: Partial<UpdateState>) => { state = { ...state, ...next }; deps.changed(state); };
  const active = () => ["checking", "downloading", "installing"].includes(state.phase);
  return {
    get state() { return state; },
    async check() {
      // Never invalidate a verified download or race another operation.
      if (active() || state.phase === "ready") return;
      set({ phase: "checking", error: null });
      try {
        const update = await deps.check();
        set({ update, phase: update ? "available" : "current" });
      } catch (e) { set({ phase: state.update ? "available" : "idle", error: String(e) }); }
    },
    async download() {
      if (state.phase !== "available") return;
      set({ phase: "downloading", error: null, downloaded: 0, total: null });
      try { await deps.download(); set({ phase: "ready" }); }
      catch (e) { set({ phase: "available", error: String(e), downloaded: 0, total: null }); }
    },
    progress(downloaded: number, total: number | null) {
      if (state.phase === "downloading") set({ downloaded, total });
    },
    async install() {
      if (state.phase !== "ready") return;
      const reason = deps.blocker();
      if (reason) { set({ error: reason }); return; }
      set({ phase: "installing", error: null });
      try { await deps.install(); }
      catch (e) { set({ phase: "ready", error: String(e) }); }
    },
  };
}
