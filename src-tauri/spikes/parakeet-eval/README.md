# Parakeet spike: accuracy, latency, linking

Measured on an M5 Max (128 GB), macOS 26.5.1, 2026-11-20. Nothing here is
wired into the app's engine list.

## What is here

- `src/main.rs` — standalone bench on crates.io `transcribe-cpp` 0.4.2
  (transcribe.cpp with its own ggml 0.26). Own `[workspace]`, so the ggml
  question cannot block it.
- `src/harness.rs` — engine-agnostic eval loop (JSONL manifest in, JSONL out),
  shared with `examples/asr_eval.rs` and `examples/three_engines.rs`.
- `../../examples/support/engines.rs` — Whisper (mirrors `stt::transcribe`),
  sherpa Parakeet (mirrors `asr::Parakeet`), whisper.cpp's `parakeet.h`, and
  transcribe.cpp on the shared ggml.
- `../fluidaudio-eval/` — FluidAudio (CoreML) runner, same output format.
- `fetch_sets.py`, `make_noise.py` — seeded subsets of public test sets with
  references (HF datasets-server / FLEURS dev tarball, streamed and cut short),
  plus synthetic/loop non-speech clips. `score.py` (WER), `latency.py`,
  `format_stats.py`, `compare.py` (byte-identical transcript check),
  `check_three.py` (same-process check), `run_matrix.sh`.

## Reproduce

```sh
W=/tmp/parakeet-work; mkdir -p $W/{data,models/tc,models/wcpp,models/fa}
python3 -m venv $W/venv && $W/venv/bin/pip install jiwer whisper-normalizer soundfile numpy
$W/venv/bin/python -I fetch_sets.py $W/data && $W/venv/bin/python -I make_noise.py $W/data
# models: handy-computer/parakeet-tdt-0.6b-v{2,3}-gguf (Q8_0) -> $W/models/tc,
#         ggml-org/parakeet-GGUF ggml-parakeet-tdt-0.6b-v3-q8_0.bin -> $W/models/wcpp
cargo build --release                                   # this crate
(cd ../.. && cargo build --release --example asr_eval --example three_engines)
(cd ../fluidaudio-eval && swift build -c release)       # optional; downloads its models
WORK=$W APP_TARGET=../../target SPIKE_TARGET=target FA_BIN=... MODELS=~/Library/Application\ Support/com.unsound.app/models \
  ./run_matrix.sh tcshared-v3 librispeech_clean ami_ihm ami_sdm ami_sdm_chunks earnings22 fleurs_de_de fleurs_fr_fr fleurs_hi_in nonspeech
$W/venv/bin/python -I score.py $W/data $W/results && $W/venv/bin/python -I latency.py $W/results
```

## Results (WER %, Whisper English normaliser; non-English: NFKC/lower/strip P,S)

| set (utts / ref words) | Whisper turbo | tc v3 | tc v2 | wcpp v3 | FA v3 ANE | FA v2 ANE | sherpa v3 |
|---|---|---|---|---|---|---|---|
| LibriSpeech test-clean (50 / 1176) | 1.36 | 1.28 | **1.11** | 1.36 | 1.36 | 1.19 | 1.96 |
| AMI IHM, ≥2 s (80 / 1140) | 12.37 | 8.60 | **7.63** | 8.77 | 7.89 | 7.63 | 22.81 (10 empty) |
| AMI SDM far-field, ≥2 s (80 / 983) | 27.16 | 20.75 | **19.23** | 20.55 | 20.24 | 19.33 | 28.08 |
| AMI SDM 20-25 s chunks (12 / 546) | 40.29 | 33.33 | **31.50** | 33.33 | 34.25 | 32.60 | 50.73 |
| Earnings-22, ≥2 s (60 / 1306) | 12.56 | **10.64** | 10.95 | 11.72 | 12.25 | 11.64 | 69.37 (31 empty) |
| FLEURS de dev (30 / 618) | **3.72** | 4.21 | n/a (en) | 4.05 | 7.28 | n/a | 10.52 |
| FLEURS fr dev (30 / 731) | 3.97 | 4.10 | n/a (en) | **3.83** | 4.65 | n/a | 16.42 |
| FLEURS hi dev (30 / 765) | **27.32** | 100 | 100 | 100 | 100 | 100 | 100 |

tc = transcribe.cpp Q8_0 GGUF; wcpp = whisper.cpp v1.9.4 `parakeet.h`, q8_0;
FA = FluidAudio CoreML, encoder on the ANE (GPU variant: same or slightly
worse). transcribe.cpp built against llama.cpp's ggml 0.24 (`tcshared`) gave
byte-identical transcripts to its own ggml 0.26 on all 379 clips, v2 and v3.
Bootstrap 95% intervals are in `score.py` output; with these sizes, gaps under
~3 points on AMI/Earnings are within noise.

Latency (warm; 237 clips of 2-10 s, 12 chunks of 20-25 s):

| engine | load ms | 1st run ms | 2-10 s p50 / p90 ms | 20-25 s p50 / max ms | peak footprint MB | on disk |
|---|---|---|---|---|---|---|
| Whisper turbo (Metal, FA) | 1080 | 275 | 274 / 312 | 417 / 878 | 2007 | 1.62 GB |
| tc v3 (shared ggml) | 122 | 78 | 44 / 58 | 105 / 225 | 354 | 740 MB |
| tc v2 (shared ggml) | 87 | 69 | 39 / 51 | 74 / 77 | 290 | 730 MB |
| wcpp v3 | 426 | 87 | 49 / 63 | 114 / 142 | 854 | 669 MB |
| FA v3 ANE | 142 | 98 | 48 / 62 | 208 / 653 | 148 | 468 MB |
| sherpa v3 (CPU int8, 2 thr) | 2234 | 1471 | 639 / 1099 | 2604 / 2999 | 2764 | 639 MB |

Load is with the model in the page cache. transcribe.cpp memory-maps weights,
so its footprint excludes them. The first run of a freshly built binary paid
a one-off Metal pipeline compile of up to ~20 s (seen for whisper.cpp
Parakeet and transcribe.cpp); later processes did not.

## Same-process check (`examples/three_engines.rs`)

Whisper turbo + Parakeet v3 + Llama 3.2 3B in one process on one ggml, 20 clips,
two rounds in opposite call order: Whisper text equals the standalone run 20/20
(and the standalone patched build equals the pre-change build 379/379), Parakeet
equals its standalone run 20/20 for both whisper.cpp's parakeet and
transcribe.cpp, LLM text is identical across rounds and to an LLM-only process
20/20.
