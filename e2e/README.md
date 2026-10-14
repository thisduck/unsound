# Meeting reliability

```sh
npm run test:reliability
npm run test:notes
```

These suites need no microphone, models, cloud credentials, or UI driver. The
notes tests use Node 24's TypeScript support. CI runs both suites and the native
backend acceptance checks on pull requests.

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

## Native backend acceptance checks (macOS)

```sh
npm run e2e:build
npm run test:native-backend
```

This runner launches the actual Tauri app with its event loop and meeting
supervisor. It needs no WebDriver key. It calls the same backend functions used
by IPC; it does not check the webview or microphone hardware. Each run uses
unique fixtures in `com.unsound.e2e`. Do not run it alongside the WebDriver suite,
which uses the same isolated profile.

The `automation` feature supplies bounded synthetic microphone chunks, a
predictable speech engine, and deterministic speech boundaries. The production
capture loop, WAV checkpointing, durable queue, result transaction, watchdog,
stop/pause/retry, and startup recovery remain in use. The speech substitute
checks the PCM actually read from disk. Normal builds contain neither these
substitutes nor the `meeting_fixture` IPC command or scenario runner.

The native scenarios check:

- A cooperative model stall trips the watchdog and leaves one retryable job.
- A non-cooperative stall retains the single worker slot while stop returns
  within 2.5 seconds and another recorder saves audio. Canceled results cannot
  commit after the stall is released.
- A meeting-scoped SQLite trigger rejects transcript inserts. Retry commits
  exactly one copy of the missing range.
- An audio append fault stops capture and reports exactly the last saved sample.
- Audio disconnection drains the final queued callback before capture closes.
- The launcher kills the app with `SIGKILL` while inference is outstanding, then
  launches two fresh processes. Committed notes and transcript survive; pending
  audio becomes exactly one transcript row; the second restart adds no rows.

Faults are in memory (SQLite triggers are temporary) and disappear on process
restart. Only a fixture's watchdog can be shortened to one second; normal
recordings keep the 90-second deadline. Commands restrict fixture IDs to safe
`e2e-` names and reject the normal application profile. The runner enforces a
60-second deadline per process and writes diagnostic logs to `e2e/artifacts/`.
The WebDriver suite exercises the same failure cases through IPC, and checks
that the user sees the storage error and can retry through the UI.

The process-kill tests are not power-loss tests. Storage faults use SQLite triggers,
read-only mode, and real OS write errors, not a physical full disk. Physical USB
removal, sleep/wake, real-model stalls, speech detection, and actual device capture
still need hardware or virtual-device validation. The synthetic source does not
measure speech recognition quality or model performance. Queue bounds do not claim a bound on
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

The specs navigate the native webview, edit notes through the UI, and check
persistence after reload and native restart. Recovery specs start synthetic
recordings through `meeting_start`; test-only IPC arms faults and supplies PCM.
They check saved sample positions, exact transcript timestamps and text, pending
counts, stop latency, visible errors, and retry. They also kill the isolated app
process while inference is outstanding. Each run deletes only its unique
fixtures. The archive stays accessible without downloaded models.

Ports 3000 and 4444 must be free. Startup has a deadline and cleans up child
services on failure. WDIO fatal setup errors prevent workers from connecting after
failed preparation. Failed specs save screenshots in `e2e/artifacts/`; binaries
and artifacts are gitignored. Native CI is a workflow-dispatch option requiring
the `CN_API_KEY_WEBDRIVER` repository secret.

Approach informed by Anarlog's [failure-oriented tests](https://github.com/fastrepl/anarlog/blob/4b8e114/crates/listener-core/src/actors/session/supervisor/reliability_tests.rs)
and [native configuration](https://github.com/fastrepl/anarlog/blob/4b8e114/e2e/blackbox/wdio.blackbox.conf.ts).
Driver setup: [CrabNebula](https://docs.crabnebula.dev/plugins/tauri-e2e-tests/) and
[Tauri WebDriver](https://v2.tauri.app/develop/tests/webdriver/).

## Notes quality and speaker corrections

```sh
npm run test:meeting-quality
LLM_MODEL=/path/to/model.gguf cargo test --locked --manifest-path src-tauri/Cargo.toml --lib synthetic_notes_on_a_real_model -- --ignored --nocapture
```

Extraction uses a grammar whose evidence choices are actual transcript lines.
Source IDs and quotations are checked again before caching. Notes composition
cites numbered facts; the app supplies their timestamps. Decisions must refer to
decision facts, and action items must refer to action facts. Unsupported owners
and due dates are left blank; an explicit first-person commitment can use its
speaker as owner. Names are substituted at word boundaries, without changing
words such as “Meeting.” The composer can remove canceled or duplicate actions.
These checks establish provenance, not semantic truth. The real-model evaluation
checks accepted tasks and deadlines, rejects treating speculation as a
commitment or decision, and removes a task canceled in a later extraction
window. These synthetic cases passed with Qwen3 4B Instruct Q4_K_M locally.
An earlier Qwen2.5 3B run missed commitments despite valid evidence. Model
quality still matters; a passing fixture is not a general accuracy guarantee.

Summary text and generated action items save in one transaction. User-edited and
completed items survive regeneration. Transcript, note, and speaker changes
advance a persisted revision. A generation based on an older revision cannot
commit; the UI marks existing notes as stale. Notes generation waits for pending
transcription to finish. Validated notes appear after composition completes.

Click a transcript speaker to correct a single utterance or the whole group.
Corrections use stable identities, retain the original audio source/time, and
survive automatic speaker re-detection and app restart. Naming a completed
speaker group pins it too. Remembered voices require at least two usable segments
and six seconds of speech. Corrected voices are rebuilt from their actual audio,
not their old cluster. Different people still require a clear similarity margin;
multiple samples of one person are treated as one identity. Offline re-detection
reads bounded audio windows instead of loading the full recording.

## Long-meeting performance (macOS)

```sh
npm run e2e:build
npm run bench:meeting -- --model /path/to/ggml-model.bin --audio /path/to/speech.wav --minutes 60 --speed 20 --restart
```

Use a 1–60 second spoken clip smaller than 10 MB and an existing local Whisper
model. The native runner repeats that clip into **both** audio channels, uses the
real ASR worker, checkpoints WAVs, and persists transcript jobs/results. It checks
that both channels retain the exact expected sample count and transcription
finishes. `--restart` kills the app after capture stops, then measures recovery in
a new process. It does not require the WebDriver key or microphone permission.
Do not run another automation-profile app at the same time.

`--speed 20` feeds twenty seconds of audio per capture poll: a 60-minute audio
workload takes roughly three minutes to feed, plus startup and queue drain.
**This is an accelerated workload, not a one-hour thermal or hardware soak.**
Use `--speed 1 --minutes 60` (or 90) for wall-clock pacing. Real device changes,
sleep/wake, VAD, and real conversational recognition quality need separate checks.

The JSON report defaults to `e2e/artifacts/meeting-benchmark.json`; select another
path with `--report`. It contains no transcript or note content. Diagnostic logs
are beside it. The fixture recording and transcript are deleted after success;
a failed run retains its unique fixture for investigation.

Measurements include:

- Successful batch processing p50/p95/max, including PCM reads, inference, speaker
  work, and result commit; real-time factor divides that time by processed audio
  across both channels. Model setup/warmup is measured separately.
- Queue-to-commit p50/p95/max, beginning when a durable job is enqueued. This is
  not microphone-to-caption latency; it excludes audio waiting inside VAD.
- WAV/checkpoint bookkeeping times and peak backlog observed by the supervisor.
- Capture stop time, drain time, and time from startup recovery scheduling until
  completion. Recovery counters start fresh in the new process.
- RSS and CPU samples from `ps` each second, peak sampled RSS, and change from the
  first sample after 30 seconds. RSS does not include all possible GPU allocations.

The same content-free diagnostics are available through `meeting_performance`.
Quantiles keep only the latest 512 samples and at most 32 meetings in memory;
maxima and totals cover each retained meeting in the current process. The JSON
grammar cache is also bounded because evidence schemas vary by transcript.

The [initial local baseline](benchmarks/2026-10-14-apple-m5-max.json) records
60 minutes per channel at 20× pacing with Whisper tiny on an Apple M5 Max:
122 ms stop, 516 ms recovery after process termination, 270 ms batch p95,
530 ms queue-to-commit p95, and about 371 MiB peak sampled RSS. Both complete
audio tracks survived, and the restarted worker drained the two remaining jobs.
Recovery finished before the first RSS sample, so its peak is unknown. Compare
like workloads and hardware; this run can benefit from cached model files.

The baseline also records the synthetic five-voice speaker evaluation. Automatic
clustering produced four speakers, with live purity 0.70 and offline purity 0.72.
This exposes a remaining quality gap. Speaker correction is tested; automatic
diarization still needs evaluation and tuning on labeled conversational audio.
