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
- **🧠 It learns your words.** Click any word it misheard to correct it — names,
  jargon, and your corrections are remembered so it gets them right next time.
- **📝 A tidy history.** Every take is kept — raw and cleaned — ready to copy,
  reopen, or re-style.
- **🔒 Completely private.** Everything runs on your machine. Works with the
  Wi-Fi off.

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

## Models

Models are downloaded on demand from Hugging Face into the app data directory
(`~/Library/Application Support/com.unsound.app/models` on macOS) and are
fully swappable per run — record once, then re-run transcription or cleanup
with different models to compare.

Curated registry: Whisper tiny / base / small / medium / large-v3-turbo, plus
Qwen 2.5 (1.5B, 3B) and Llama 3.2 (1B, 3B) instruct models at Q4_K_M. Any
other model works via **models → add a custom model by URL** — whisper.cpp
GGML files for speech, llama.cpp GGUF files for cleanup.

The cleanup system prompt is editable in the app (panel 03 → edit prompt).

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
- [ ] Per-app automatic styles (informal in Slack, professional in Mail)
- [ ] Windows build (same codebase; CPU inference by default, Vulkan optional)
- [ ] iOS / Android — whisper.cpp and llama.cpp both run on mobile; the plan
      is to reuse the pipeline design and model registry, with a native or
      React Native UI (`whisper.rn` / `llama.rn`)
- [ ] Download resume + checksum verification
