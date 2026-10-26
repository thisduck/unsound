import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getIdentifier, getVersion } from "@tauri-apps/api/app";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Markdown } from "./Markdown";
import { createUpdateController, type AppUpdate } from "./updateController";
import { updateWorkBlocker } from "./updateActivity";
import { formatBytes } from "./api";

const AUTO_KEY = "unsound.auto-check-updates";
export const openUpdates = () => window.dispatchEvent(new Event("unsound:check-updates"));

export function UpdateCenter({ busy }: { busy: boolean }) {
  const busyRef = useRef(busy); busyRef.current = busy;
  const [visible, setVisible] = useState(false);
  const [version, setVersion] = useState("");
  const [releaseApp, setReleaseApp] = useState(false);
  const [automatic, setAutomatic] = useState(() => {
    try { return localStorage.getItem(AUTO_KEY) !== "false"; } catch { return true; }
  });
  const [nativeBlocker, setNativeBlocker] = useState<string | null>(null);
  const [preferenceError, setPreferenceError] = useState<string | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  const controllerRef = useRef<ReturnType<typeof createUpdateController> | null>(null);
  const [, render] = useState(0);
  if (!controllerRef.current) controllerRef.current = createUpdateController({
    check: () => invoke<AppUpdate | null>("check_app_update"),
    download: () => invoke("download_app_update"),
    install: async () => {
      const reason = await invoke<string | null>("app_update_blocker");
      if (reason) throw new Error(reason);
      // Recheck after the IPC round trip: a hotkey may have started a take.
      const pending = busyRef.current ? "Finish dictation before restarting to update." : updateWorkBlocker();
      if (pending) throw new Error(pending);
      await invoke("install_app_update");
    },
    blocker: () => busyRef.current ? "Finish dictation before restarting to update." : updateWorkBlocker(),
    changed: () => render(n => n + 1),
  });
  const controller = controllerRef.current;
  const state = controller.state;
  const locked = state.phase === "installing";
  const inProgress = ["checking", "downloading", "installing"].includes(state.phase);

  useEffect(() => {
    void getVersion().then(setVersion).catch(() => {});
    void getIdentifier().then(id => setReleaseApp(id === "com.unsound.app")).catch(() => {});
    const manual = () => { setVisible(true); void controller.check(); };
    window.addEventListener("unsound:check-updates", manual);
    const subscriptions = [
      listen("check-for-updates", manual),
      listen<{ downloaded: number; total: number | null }>("app-update-progress", e => controller.progress(e.payload.downloaded, e.payload.total)),
    ];
    return () => {
      window.removeEventListener("unsound:check-updates", manual);
      subscriptions.forEach(p => { void p.then(unlisten => unlisten()).catch(() => {}); });
    };
  }, [controller]);

  useEffect(() => {
    // Development and test builds never make unsolicited update requests.
    if (!automatic || !releaseApp || import.meta.env.DEV) return;
    const initial = setTimeout(() => { void controller.check(); }, 1500);
    const interval = setInterval(() => { void controller.check(); }, 12 * 60 * 60 * 1000);
    return () => { clearTimeout(initial); clearInterval(interval); };
  }, [automatic, releaseApp, controller]);

  useEffect(() => {
    if (!visible) return;
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.showModal();
    let active = true;
    const poll = () => {
      void invoke<string | null>("app_update_blocker").then(reason => {
        if (active) setNativeBlocker(reason);
      }).catch(() => { if (active) setNativeBlocker("Could not check recording status. Try again shortly."); });
    };
    poll();
    const timer = setInterval(() => { poll(); render(n => n + 1); }, 1000);
    return () => { active = false; clearInterval(timer); previous?.focus(); };
  }, [visible]);

  const blocker = busy ? "Finish dictation before restarting to update." : updateWorkBlocker() ?? nativeBlocker;
  const hasUpdate = !!state.update;
  const progress = state.total ? Math.min(100, Math.round(state.downloaded / state.total * 100)) : undefined;
  const close = () => { if (!locked) setVisible(false); };
  return <>
    <button className={`quiet${hasUpdate ? " update-available" : ""}`} onClick={() => { setVisible(true); if (!hasUpdate) void controller.check(); }}>
      {hasUpdate ? `update ${state.update!.version}` : "updates"}
    </button>
    {visible && createPortal(<dialog ref={dialog} className="update-dialog" aria-labelledby="update-title" onCancel={e => { e.preventDefault(); close(); }}>
      <header className="sheet-head">
        <h2 id="update-title">{hasUpdate ? `unsound ${state.update!.version}` : "Software updates"}</h2>
        <button className="quiet" disabled={locked} onClick={close} aria-label="Close updates">close ✕</button>
      </header>
      <div className="update-body">
        <p className="sheet-hint">Installed version {version || state.update?.currentVersion || "…"}</p>
        <p role="status" aria-live="polite">{
          state.phase === "checking" ? "Checking for updates…" :
          state.phase === "current" ? "You’re up to date." :
          state.phase === "downloading" ? `Downloading update… ${progress === undefined ? formatBytes(state.downloaded) : `${progress}%`}` :
          state.phase === "ready" ? "Update downloaded and verified. Restart when you’re ready." :
          state.phase === "installing" ? "Installing update and restarting…" :
          hasUpdate ? "A new version of unsound is available." : "Check for the latest version of unsound."
        }</p>
        {state.phase === "downloading" && <progress aria-label="Update download" value={progress} max={100} />}
        {hasUpdate && <section className="update-notes" aria-label="Release notes">
          <h3>What’s new</h3>
          <Markdown text={state.update!.notes || "See the full release notes for details."} />
        </section>}
        <button className="quiet" disabled={locked} onClick={() => { void import("@tauri-apps/plugin-opener").then(({ openUrl }) => openUrl("https://github.com/thisduck/unsound/releases")).catch(e => setPreferenceError(String(e))); }}>View releases ↗</button>
        {state.error && <p className="sheet-error" role="alert">Could not complete the update: {state.error}</p>}
        {blocker && state.phase === "ready" && <p role="status">{blocker}</p>}
        <div className="update-actions">
          {state.phase === "ready" || locked ? <button className="primary" disabled={locked || !!blocker} onClick={() => { void controller.install(); }}>Update &amp; restart</button> :
            hasUpdate ? <button className="primary" disabled={inProgress} onClick={() => { void controller.download(); }}>{state.phase === "downloading" ? "Downloading…" : "Download update"}</button> :
            <button className="primary" disabled={inProgress} onClick={() => { void controller.check(); }}>Check for updates</button>}
          <button className="quiet" disabled={locked} onClick={close}>{hasUpdate ? "Later" : "Close"}</button>
        </div>
        <label className="update-preference"><input type="checkbox" checked={automatic} disabled={locked} onChange={e => {
          const enabled = e.target.checked;
          setAutomatic(enabled); setPreferenceError(null);
          try { localStorage.setItem(AUTO_KEY, String(enabled)); } catch { setPreferenceError("Could not save this preference. It applies until you quit unsound."); }
        }} /> Automatically check for updates</label>
        <p className="sheet-hint">Checks GitHub on launch and every 12 hours. Your recordings and notes are never sent. Downloads and restarts happen only when you choose.</p>
        {preferenceError && <p className="sheet-error" role="alert">{preferenceError}</p>}
      </div>
    </dialog>, document.body)}
  </>;
}
