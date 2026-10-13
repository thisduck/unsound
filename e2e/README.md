# Reliability checks

Run the deterministic fault suite without models, microphone permissions, provider
credentials, or the native UI driver:

```sh
npm run test:reliability
```

The tests use disposable directories, the production SQLite/WAV code, and narrow
injection points in the recorder and transcription queue. The crash test launches
a separate test process, waits until it has written its fixtures, kills it without
cleanup, and opens the files again through the app's recovery functions.

| Failure | What the test checks |
| --- | --- |
| Process killed with SQLite and WAV open | Committed notes and transcript survive; an uncommitted update does not. All 48,000 written PCM samples survive despite the stale WAV header. Recovery is repeatable. |
| Storage rejects the second insert in a batch | The entire batch rolls back; previously saved transcript survives; an explicit retry writes exactly one copy. |
| Database becomes read-only | Updates return errors and leave the last saved notes intact after reopening. Writes work after storage is restored. |
| WAV writes/finalization fail | Errors reach the caller, failed samples are not counted, and already written audio can be repaired. |
| WAV ends with half a sample | Repair removes the incomplete sample, preserves complete samples, and is repeatable. |
| Partial inference stalls | The production queue reservation stays held during blocked inference. More previews are refused, their payload is capped at 25 seconds, and independent WAV writes continue. |
| Inference consumer disconnects | A rejected preview releases its queue reservation. |
| Audio producer disconnects | Already buffered audio and the tail drain once; stop releases the recorder; an active dictation tap keeps its audio. |
| Previous session still finalizes | Capture retains the recorder slot until it stops. A new session can occupy it while old inference finishes; the old completion cannot clear the new recording flag. |
| Provider rejects credentials | A local HTTP server receives one request with the expired key, then a new request with the replacement key. A 401 does not trigger format-fallback retries. |

These checks deliberately separate existing guarantees from missing ones:

- Final transcription jobs still use an unbounded queue. The bound above applies
  to previews, not the complete pipeline. A durable queue with a memory budget is
  still needed before promising bounded memory during a long inference stall.
- Inference cannot yet be canceled or timed out reliably. Stopping a meeting can
  wait indefinitely for a stuck model. These tests inject a controlled stall at
  the queue boundary; they do not hang a real model runtime.
- Startup preserves audio and committed transcript, but does not re-transcribe
  audio that had not reached the database. The crash test explicitly checks this.
- Capture still ignores several storage errors. Writer tests prove errors are
  returned, not that the UI warns or that capture retries them. SQLite failures
  here use an abort trigger and read-only mode, not a physical full disk.
- Audio tests disconnect the producer channels. Physical USB removal, macOS
  ScreenCaptureKit interruption, sleep/wake, and device reconnection still need a
  virtual-device or hardware test environment.
- The credential test supplies a replacement key on the next request. It does
  not claim automatic token refresh or exercise the user's Keychain.
- Old-session UI events and shared model contention during overlapping
  finalization still need end-to-end coverage.

# Native Tauri UI tests (macOS)

```sh
npm ci
npm run e2e:build
# Supply CN_API_KEY_WEBDRIVER through your local environment or CI secret store.
npm run test:e2e
```

The harness uses WebdriverIO, `tauri-driver`, and CrabNebula's test runner backend.
`npm run e2e:check` checks the key and the built binary without launching services.
A missing prerequisite fails the command with a nonzero exit code; no tests are
silently skipped. The build itself does not require the key.

`e2e:build` enables the optional `automation` Cargo feature and uses
`src-tauri/tauri.e2e.conf.json`. It copies the result to `e2e/bin/unsound-e2e`.
The app refuses to start an automation build with any identifier other than
`com.unsound.e2e`. App data and Keychain entries therefore use a separate namespace.
Global hotkeys, the tray, background assistant, and meeting detection are not
started. The test does not record audio or download models. Normal builds do not
include the automation plugin.

The spec navigates the actual native webview, creates synthetic meeting data over
real Tauri IPC, checks persistence after a webview reload, then creates a new
WebDriver session and checks native startup recovery. Metadata is written through
IPC because the meeting screen requires downloaded models; this is not yet a UI
test of recording or note editing. Each run uses a unique meeting ID and deletes
only that fixture. The isolated app profile remains available for inspection.

Ports 3000 and 4444 must be free. Service startup has a 30-second deadline; failures
and shutdown clean up the child services. Failed specs save screenshots under
`e2e/artifacts/`. Both binaries and artifacts are gitignored.

The reliability workflow runs the Rust suite and frontend build on pull requests.
Native UI tests are an explicit workflow-dispatch option and need the
`CN_API_KEY_WEBDRIVER` repository secret. Do not treat a successful build or
preflight as a completed native UI test run.

Approach informed by Anarlog's [failure-oriented tests](https://github.com/fastrepl/anarlog/blob/4b8e114/crates/listener-core/src/actors/session/supervisor/reliability_tests.rs)
and [native test configuration](https://github.com/fastrepl/anarlog/blob/4b8e114/e2e/blackbox/wdio.blackbox.conf.ts).
Setup references: [CrabNebula](https://docs.crabnebula.dev/plugins/tauri-e2e-tests/)
and [Tauri WebDriver](https://v2.tauri.app/develop/tests/webdriver/).
