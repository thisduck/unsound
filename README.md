# unsound

Local dictation and cleanup. Record your voice, get a transcript from a local
Whisper model, then have a local LLM tidy it up — punctuation, filler words,
paragraphs. **Nothing ever leaves your machine.** The only network access in
the entire app is the model downloader; every other feature works with the
network cable unplugged.

## How it works

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
- [ ] Windows build (same codebase; CPU inference by default, Vulkan optional)
- [ ] iOS / Android — whisper.cpp and llama.cpp both run on mobile; the plan
      is to reuse the pipeline design and model registry, with a native or
      React Native UI (`whisper.rn` / `llama.rn`)
- [ ] Download resume + checksum verification
