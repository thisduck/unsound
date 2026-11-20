// macOS permissions UI: the first-run "Allow access" step, Settings ›
// Permissions, and the one-time check for existing users who are missing a
// required permission. Decisions live in permissionsModel.ts.
import { ReactNode, useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { api } from "./api";
import {
  actionFor,
  addKinds,
  ASKED_KEY,
  detailFor,
  missingRequired,
  needsRelaunch,
  needsReview,
  OPEN_PERMISSIONS_EVENT,
  PERMISSIONS,
  readKinds,
  REVIEWED_KEY,
  statusLabel,
  type Access,
  type PermissionDef,
  type PermissionEntry,
  type PermissionKind,
} from "./permissionsModel";
import { PrefCard, PrefGroup } from "./sections";
import { Icon } from "./ui";

export const permissionsApi = {
  all: () => invoke<PermissionEntry[]>("permissions_all"),
  request: (kind: PermissionKind) => invoke<Access>("permissions_request", { kind }),
  openSettings: (kind: PermissionKind) => invoke<void>("permissions_open_settings", { kind }),
  relaunch: () => invoke<void>("relaunch_app"),
};

export function useOpenPermissions(cb: () => void) {
  useEffect(() => {
    window.addEventListener(OPEN_PERMISSIONS_EVENT, cb);
    return () => window.removeEventListener(OPEN_PERMISSIONS_EVENT, cb);
  }, [cb]);
}

const storage = () => {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage;
  } catch {
    return undefined;
  }
};

/** Live permission status: refreshed on a timer while mounted and whenever
 *  the window regains focus (people flip a switch in System Settings and come back). */
export function usePermissions(live = true) {
  const [entries, setEntries] = useState<PermissionEntry[] | null>(null);
  const refresh = useCallback(
    () =>
      permissionsApi
        .all()
        .then(setEntries)
        .catch(() => {}),
    [],
  );
  useEffect(() => {
    refresh();
    if (!live) return;
    const timer = window.setInterval(refresh, 1500);
    window.addEventListener("focus", refresh);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh, live]);
  return { entries, refresh };
}

type Variant = "onboarding" | "settings";

function PermissionGroups({
  entries,
  refresh,
  variant,
  onError,
}: {
  entries: PermissionEntry[] | null;
  refresh: () => Promise<unknown>;
  variant: Variant;
  onError: (msg: string | null) => void;
}) {
  const [asked, setAsked] = useState<PermissionKind[]>(() => readKinds(storage(), ASKED_KEY));
  const [busy, setBusy] = useState<PermissionKind | null>(null);
  const [mic, setMic] = useState<"idle" | "checking" | "ok">("idle");
  const defs = PERMISSIONS.filter((d) => variant === "settings" || d.onboarding);
  const entry = (kind: PermissionKind) => entries?.find((e) => e.kind === kind);
  const relaunch = entries ? needsRelaunch(entries, asked) : [];

  const act = async (def: PermissionDef, action: "allow" | "settings" | "enable") => {
    onError(null);
    setBusy(def.kind);
    try {
      if (action !== "enable") setAsked(addKinds(storage(), ASKED_KEY, [def.kind]));
      if (action === "settings") await permissionsApi.openSettings(def.kind);
      else await permissionsApi.request(def.kind);
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(null);
      await refresh();
    }
  };

  const testMic = async () => {
    onError(null);
    setMic("checking");
    try {
      await api.requestMicrophone();
      setMic("ok");
    } catch (e) {
      setMic("idle");
      onError(String(e));
    }
  };

  const row = (def: PermissionDef) => {
    const e = entry(def.kind);
    const action = actionFor(e, asked.includes(def.kind));
    const status = statusLabel(def, e);
    const waiting = busy === def.kind;
    let control = null;
    if (action !== "none") {
      control = (
        <button
          className={"btn btn-sm " + (action === "settings" ? "btn-secondary" : "btn-primary")}
          disabled={busy !== null}
          aria-busy={waiting}
          onClick={() => act(def, action)}
        >
          {waiting ? "Waiting…" : action === "allow" ? "Allow…" : action === "enable" ? "Turn on" : "Open Settings"}
        </button>
      );
    } else if (variant === "settings" && def.kind === "microphone" && e?.access === "granted") {
      control =
        mic === "ok" ? (
          <span className="perm-granted">
            <Icon name="check" /> Working
          </span>
        ) : (
          <button className="btn btn-secondary btn-sm" onClick={testMic} disabled={mic === "checking"}>
            {mic === "checking" ? "Listening…" : "Test microphone"}
          </button>
        );
    } else if (variant === "onboarding" && e?.access === "granted") {
      control = (
        <span className="perm-granted">
          <Icon name="check" /> Allowed
        </span>
      );
    }
    return (
      <div className="pref-row perm-row" key={def.kind} data-permission={def.kind}>
        <span className="perm-icon" aria-hidden="true">
          <Icon name={def.icon} />
        </span>
        <div className="pref-text">
          <div className="pref-label">
            {def.title}
            {variant === "onboarding" && !def.required && <span className="badge neutral">Optional</span>}
            {variant === "settings" && status && <span className={"badge " + status.tone}>{status.text}</span>}
          </div>
          <div className="pref-desc">{detailFor(def, e)}</div>
        </div>
        {control && <div className="pref-control">{control}</div>}
      </div>
    );
  };

  const required = defs.filter((d) => d.required);
  const optional = defs.filter((d) => !d.required);
  return (
    <>
      <PrefGroup title="Needed for dictation">
        <PrefCard>{required.map(row)}</PrefCard>
      </PrefGroup>
      <PrefGroup
        title="Optional"
        hint={variant === "settings" ? "Each one turns on a feature. unsound works without them." : "For meetings. Skip any you don't need."}
      >
        <PrefCard>{optional.map(row)}</PrefCard>
      </PrefGroup>
      {relaunch.length > 0 && (
        <div className="alert perm-relaunch" role="status">
          <Icon name="info" />
          <div className="alert-body">
            After you turn on unsound under <b>Screen &amp; System Audio Recording</b>, macOS needs unsound to reopen.
          </div>
          <div className="alert-actions">
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => permissionsApi.relaunch().catch((e) => onError(String(e)))}
            >
              <Icon name="refresh" /> Relaunch unsound
            </button>
          </div>
        </div>
      )}
    </>
  );
}

/** Settings › Permissions. */
export function PermissionsPane({ onError }: { onError: (msg: string | null) => void }) {
  const { entries, refresh } = usePermissions();
  return (
    <>
      <PermissionGroups entries={entries} refresh={refresh} variant="settings" onError={onError} />
      <p className="pref-note">
        Changes made in System Settings show up here when you come back. Turning a permission off in System Settings is
        the only way to revoke it.
      </p>
    </>
  );
}

/** The "Allow access" step of first-run onboarding. */
export function PermissionsStep({
  onNext,
  onError,
  children,
}: {
  onNext: () => void;
  onError: (msg: string | null) => void;
  children?: ReactNode;
}) {
  const { entries, refresh } = usePermissions();
  const ready = !!entries && missingRequired(entries).length === 0;
  useEffect(() => {
    // Seen during setup; the launch-time check won't repeat these.
    addKinds(storage(), REVIEWED_KEY, ["microphone", "accessibility"]);
  }, []);
  return (
    <div className="ob-step">
      <h1 className="ob-title">Allow access</h1>
      <p className="ob-text">
        macOS asks once for each. Choose <b>Allow</b> to see its prompt, or open System Settings if you said no before.
      </p>
      <PermissionGroups entries={entries} refresh={refresh} variant="onboarding" onError={onError} />
      <div className="ob-actions">
        <button className="btn btn-primary ob-primary" disabled={!ready} onClick={onNext}>
          Continue
        </button>
        {!ready && (
          <button className="btn btn-quiet" onClick={onNext}>
            Set up later
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

/**
 * For people who finished setup before this step existed, or skipped it: at
 * launch, if a required permission is off and they haven't been shown it,
 * show the access step once. Closing it is remembered per permission.
 */
export function PermissionsCheck() {
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const { entries, refresh } = usePermissions(open);

  useEffect(() => {
    if (!open && entries && needsReview(true, readKinds(storage(), REVIEWED_KEY), entries)) setOpen(true);
    // Only the first answer decides; later polls just update the rows.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries === null]);

  if (!open) return null;
  const close = () => {
    addKinds(storage(), REVIEWED_KEY, entries ? missingRequired(entries) : []);
    setOpen(false);
  };
  const ready = !!entries && missingRequired(entries).length === 0;
  return (
    <div className="ob-overlay">
      <div className="ob-frame" role="dialog" aria-modal="true" aria-labelledby="perm-check-title">
        <div className="ob-body">
          <div className="ob-step">
            <h1 className="ob-title" id="perm-check-title">
              {ready ? "You're all set" : "unsound needs access"}
            </h1>
            <p className="ob-text">
              {ready
                ? "Dictation has everything it needs."
                : "Dictation needs these to hear you and type into other apps. You can change them any time in Settings › Permissions."}
            </p>
            <PermissionGroups entries={entries} refresh={refresh} variant="onboarding" onError={setErr} />
            <div className="ob-actions">
              <button className={"btn ob-primary " + (ready ? "btn-primary" : "btn-secondary")} onClick={close}>
                {ready ? "Done" : "Not now"}
              </button>
            </div>
            {err && (
              <div className="alert danger" role="alert">
                <Icon name="alert" />
                <div className="alert-body">{err}</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
