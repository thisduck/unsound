# Meeting reliability

```sh
npm run test:reliability
npm run test:notes
```

These suites need no microphone, models, cloud credentials, or UI driver. The
notes tests use Node 24's TypeScript support. CI runs both suites on pull requests.

## Saved audio and durable work

Capture checkpoints each channel's WAV before publishing its sample ranges to
SQLite. Final transcription jobs contain sample offsets, not audio buffers. One
meeting transcription worker reads at most 25 seconds per batch, plus leading silence; its disposable
preview/command channel holds at most four jobs, each capped at 25 seconds. Long
backlogs stay on disk. Whisper still batches short turns from the same channel;
all completion markers in a batch commit with its transcript. Silence padding uses a fixed-size buffer.

Transcript rows and their job completion marker commit in one transaction. A
failed commit leaves the job pending. Successfully recognized silence completes a
job too. Inference errors no longer masquerade as successful empty transcripts.

Stopping releases capture without joining inference. The UI shows “finishing
transcript” while queued audio is processed, and allows another recording. Final
results, previews, and speaker events carry meeting IDs so an older session cannot
update the new meeting's transcript. Summaries wait for transcript completion.

Startup repairs unfinished WAV headers and queues uncovered ranges, including the
last utterance that was still inside VAD. Completed ranges and persisted
command/dictation exclusions are not replayed. Recovery does not execute spoken
assistant requests from an earlier session. Recovered speaker labels are generic
rather than overwriting previously assigned speakers.

This recovery applies to meetings recorded with the durable queue. Older meetings
without a `meeting_work` row are not automatically re-transcribed.

## Failures and controls

Audio/queue storage failures stop capture and retain an error in the UI. Status
shows each channel's last saved position. Errors are persisted when SQLite remains
writable, with an in-memory fallback when it does not. The warning records a prior
interruption after retry; recovery cannot restore audio that never reached disk.

An interrupted microphone reports an error instead of only logging it. Shutdown
drains the microphone after its last callback and preserves the system-audio tail.
A long ScreenCaptureKit gap reports an interruption rather than allocating a large
silence buffer in memory.

“Pause” prevents an in-flight result from committing and requests native
cancellation. The supervisor also cancels after 90 seconds without progress.
Whisper has an abort callback; other native model calls may need to return on their
own. A stuck call keeps the single meeting-worker slot, so retry does not create
unbounded replacement threads. If it cannot stop, restart the app; pending work
remains on disk. Capture and its stop control do not wait for that inference call.

Notes edits retain a local draft, report failed database saves, and flush when an
editor closes. Writes from the live and archive editors run in order. Local draft
storage failures are visible too; the user must keep the window open until a
successful database save in that case.

## Fault coverage

| Injection | Assertion |
| --- | --- |
| Kill a separate process with SQLite/WAV open | Committed notes/transcript survive; an uncommitted update rolls back; queued audio and the unqueued VAD tail replay exactly once. |
| Fail the second transcript insert | Neither partial transcript nor a completion marker commits; retry writes one copy. |
| Pause while a result is outstanding | The late result cannot commit; its job remains pending for retry. |
| Reopen with completed work and private tap intervals | Recovery fills only uncovered audio ranges, including sub-window tails. |
| Create a thousand-job backlog | Each job remains bounded; completed work leaves the pending queue. |
| Make SQLite read-only | Saves return errors and previously saved notes survive reopening. |
| Fail WAV writes/finalization; tear a PCM sample | Saved samples remain readable; repair removes the partial sample and is repeatable. |
| Stall a preview; disconnect its consumer | Capture writes continue; previews do not accumulate; failed enqueue releases its reservation. |
| Disconnect the audio producer; deliver a final callback during shutdown | Buffered audio drains once, including the shutdown tail and active dictation taps. |
| Start another session while old inference finalizes | Capture ownership is released first; old completion cannot clear the new recording flag. |
| Resume system audio after a long gap | The gap is reported without allocating the whole missing interval. |
| Reject credentials, then supply a replacement | The local HTTP server receives the new key; 401 does not trigger format-fallback retries. |
| Fail notes saving; edit during an outstanding save; switch editors | Draft survives and older completion cannot erase or overwrite a newer edit. |

The process-kill test is not a power-loss test. Storage faults use SQLite triggers,
read-only mode, and real OS write errors, not a physical full disk. Physical USB
removal, sleep/wake, real-model stalls, and the full native recording flow still
need hardware or virtual-device validation. Queue bounds do not claim a bound on
model memory or every microphone tap buffer. A blocked OS storage call can still
delay capture shutdown.

## Native Tauri UI tests (macOS)

```sh
npm ci
npm run e2e:build
# Supply CN_API_KEY_WEBDRIVER through the environment or a CI secret.
npm run test:e2e
```

The harness uses WebdriverIO, `tauri-driver`, and CrabNebula's test runner backend.
`npm run e2e:check` checks prerequisites without launching services. Missing
credentials fail the command; tests are not silently skipped. The build itself
does not need a key.

The optional `automation` feature and `src-tauri/tauri.e2e.conf.json` produce
`e2e/bin/unsound-e2e`. Automation builds refuse the normal app identifier: app data
and Keychain entries use `com.unsound.e2e`. Global hotkeys, tray, meeting detection,
and background assistants are disabled. Normal builds omit the automation plugin.

The spec navigates the native webview, creates a synthetic meeting through real
IPC, edits notes through the UI, and checks persistence after webview reload and
native session restart. The archive stays accessible without downloaded models.
It does not yet record a real meeting. Each run deletes only its unique fixture.

Ports 3000 and 4444 must be free. Startup has a deadline and cleans up child
services on failure. WDIO fatal setup errors prevent workers from connecting after
failed preparation. Failed specs save screenshots in `e2e/artifacts/`; binaries
and artifacts are gitignored. Native CI is a workflow-dispatch option requiring
the `CN_API_KEY_WEBDRIVER` repository secret.

Approach informed by Anarlog's [failure-oriented tests](https://github.com/fastrepl/anarlog/blob/4b8e114/crates/listener-core/src/actors/session/supervisor/reliability_tests.rs)
and [native configuration](https://github.com/fastrepl/anarlog/blob/4b8e114/e2e/blackbox/wdio.blackbox.conf.ts).
Driver setup: [CrabNebula](https://docs.crabnebula.dev/plugins/tauri-e2e-tests/) and
[Tauri WebDriver](https://v2.tauri.app/develop/tests/webdriver/).
