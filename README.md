# unsound

**Talk, and unsound turns it into clean writing — entirely on your own Mac.**

Speak into unsound and it writes down what you said, then quietly tidies it
up: fixes punctuation, drops the "um"s and false starts, breaks it into
paragraphs. Nothing you say ever leaves your computer — no cloud, no account,
no internet. The only time it touches the network is the one-time download of
the models that do the work.

[**Download the latest release →**](https://github.com/thisduck/unsound/releases)
(macOS, Apple Silicon)

### What you can do with it

- **🎙 Dictate anywhere.** Press a keyboard shortcut in any app — Mail, Slack,
  Notes, your browser — speak, and the cleaned-up text is typed right where
  your cursor is. Two modes: press once to start and stop, or hold-to-talk.
- **✍️ Write in your own voice.** Teach unsound your writing styles by pasting
  a few samples ("casual", "professional", …). It matches your tone,
  punctuation, even lowercase habits — and you can switch styles with a click.
- **📁 Transcribe audio files.** Drop in a voice memo, a meeting recording, or
  a WhatsApp voice note (even ones that downloaded with the wrong name) and get
  a clean transcript.
- **🌍 Speak any language, get any language.** Transcribe ~99 languages, then
  keep the original, transliterate it to the Latin alphabet, or translate it to
  another language — all at the click of a dropdown.
- **🧠 It learns your words.** Click any word it misheard to correct it — names,
  jargon, and your corrections are remembered so it gets them right next time.
- **📝 A tidy history.** Every take is kept — raw and cleaned — ready to copy,
  reopen, or re-style.
- **🔎 Comfortable to read.** Adjustable text size for the transcript and
  refined output.
- **🗓 Meetings.** Start one (or let unsound offer when Zoom, Meet or Teams
  takes your mic) and it transcribes you and the call live, tells the other
  speakers apart as they talk — and recognizes people you've named before —
  and writes notes built around the ones you jot, with action items, open
  questions, and timestamps you can click to hear that moment again.
- **🙋 Ask it to do things.** Mid-meeting, say "unsound, create a Jira ticket
  from what we just discussed" or "…draft an email to Priya about the launch
  date" (or hold your command key — it works while you're muted on the
  call). It drafts from the conversation right away; you review, edit and
  approve before anything is sent. Mail, Reminders, Calendar, Jira, any MCP
  tool server, or a hand-off to Claude Code.
- **🌐 Use the sites you're already signed into.** Describe your tools in your
  own words — "Jira: tickets; if no project is mentioned use BLAH" — and
  unsound follows your rules and does the work in your own browser (through
  Claude in Chrome, or its own model with a browser connector). No
  integration needed per site.
- **🔎 Search every meeting.** Full-text search across all transcripts, or ask
  a question across meetings and get an answer with sources.
- **🔒 Completely private.** Everything runs on your machine. Works with the
  Wi-Fi off — the only exceptions are tools you connect and approve.

Pick the models you like from a built-in library (smaller and faster, or
larger and sharper) — they download once and you can swap them anytime.

---

## How it works (for the technically curious)

```
microphone ──cpal──▶ 16kHz mono PCM ──whisper.cpp──▶ raw transcript
                                                          │
                                     llama.cpp ◀──────────┘
                                 (cleanup prompt)
                                          │
                                          ▼
                                    refined text
```

- **Shell:** [Tauri 2](https://tauri.app) — Rust backend, React + TypeScript front end.
- **Speech-to-text:** [whisper.cpp](https://github.com/ggml-org/whisper.cpp) via `whisper-rs`, Metal-accelerated on macOS.
- **Cleanup LLM:** [llama.cpp](https://github.com/ggml-org/llama.cpp) via `llama-cpp-2`, Metal-accelerated on macOS. Runs any GGUF instruct model.
- **Audio capture:** `cpal` (CoreAudio / WASAPI), downmixed and resampled to the 16 kHz mono Whisper expects with `rubato`.

### Meetings, in more detail

```
mic ─┐                      ┌─▶ whisper / parakeet ─▶ live transcript ─┐
     ├─▶ disk (both WAVs) ──┤   (commands jump the queue; wake word    │
sys ─┘   + Silero VAD       │    lifted into the assistant)            │
                            └─▶ voice embeddings ─▶ live speaker labels │
                                                                        ▼
        every 5 minutes, while you talk:  extract grounded points (LLM) │
        after transcription: compose → validate → save notes + actions ◀┘
```

- **Audio** from both channels streams to `meeting-audio/<id>.wav` and
  `<id>.mic.wav` as it's captured, on one clock (the mic's); system audio is
  placed by its buffer timestamps so silent stretches don't drift it.
- **Speakers** are clustered live from CAM++ voice embeddings; tiny phantom
  clusters are merged at the end. Correct individual utterances or whole groups;
  corrections survive re-detection. Clear speech samples can be remembered.
- **Notes** are extracted per five-minute stretch during the meeting and
  cached by the exact text they came from, so edits and renames only redo
  what changed. Notes and action items cite transcript sources and save together
  after validation. Later edits mark the notes as needing regeneration.
- **The assistant** turns a request into a task: plan (what, when, about which
  part of the meeting) → draft (grammar-constrained JSON, so drafts are always
  well-formed) → your approval → run. Only your mic can give requests, and the
  transcript is always handled as untrusted data. Pick a stronger local model
  for it (Gemma 4 12B/26B-A4B, Qwen3.6, gpt-oss) or any OpenAI-compatible server.

## Models

Models are downloaded on demand from Hugging Face into the app data directory
(`~/Library/Application Support/com.unsound.app/models` on macOS) and are
fully swappable per run — record once, then re-run transcription or cleanup
with different models to compare.

Curated registry:

- **Speech-to-text (Whisper):** tiny / base / small / medium / large-v3-turbo /
  large-v3, English-only base & small variants, plus fine-tunes for South Asian
  languages — IndicWhisper Medium (AI4Bharat) and Whisper Hindi→Hinglish.
- **Cleanup (llama.cpp GGUF):** Qwen3 4B (default), Gemma 3 4B, Qwen 2.5
  (1.5B/3B/7B), Llama 3.1 8B, Llama 3.2 (1B/3B).

Any other model works via **models → add a custom model by URL** — whisper.cpp
GGML files for speech, llama.cpp GGUF files for cleanup.

The cleanup system prompt is read-only in the app, with a box for your own
additions on top of it (settings → prompt).

## Installing a release

Grab the `.dmg` from [Releases](https://github.com/thisduck/unsound/releases),
drag unsound to Applications, and open it — builds are code-signed and
notarized.

## Development

Prereqs: Rust, Node 20+, cmake (`brew install cmake`).

```sh
npm install
npm run tauri dev      # run the app
npm run tauri build    # produce a bundled .app / .dmg
```

The first build compiles whisper.cpp and llama.cpp from source and takes a few
minutes; after that builds are incremental.

Note for dev mode: macOS attributes the microphone permission to your
terminal. The bundled app has its own `NSMicrophoneUsageDescription`
(see `src-tauri/Info.plist`).

## Roadmap

- [x] macOS desktop app
- [x] Global shortcuts — hands-free and push-to-talk (multiple bindings,
      fn-key support), typing the refined text into any app, with a floating
      waveform overlay
- [x] History of past takes (raw + refined, local only)
- [x] Menu bar tray, microphone switching, onboarding
- [x] Signed + notarized releases from CI
- [x] Writing styles from your own samples; quick style switching
- [x] Personal dictionary — click-to-correct that biases recognition + cleanup
- [x] Audio file upload (drag-drop or picker): wav/mp3/m4a/flac/ogg, Ogg-Opus
      and fragmented-MP4 WhatsApp voice notes, content-sniffed by bytes
- [x] Per-take output language: keep, transliterate to Latin script, or
      translate to another language (at the cleanup stage)
- [x] Adjustable text size (accessibility)
- [ ] Per-app automatic styles (informal in Slack, professional in Mail)
- [ ] A second STT engine (sherpa-onnx) for Parakeet/Moonshine and better
      Urdu/Punjabi coverage
- [ ] Windows build (same codebase; CPU inference by default, Vulkan optional)
- [ ] iOS / Android — whisper.cpp and llama.cpp both run on mobile; the plan
      is to reuse the pipeline design and model registry, with a native or
      React Native UI (`whisper.rn` / `llama.rn`)
- [ ] Download resume + checksum verification
