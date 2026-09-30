# whisper-rs-sys (patched)

whisper-rs-sys 0.15.0 with two changes, wired in by `[patch.crates-io]` in
`src-tauri/Cargo.toml`:

1. **One ggml.** whisper.cpp and llama.cpp each bundle ggml, the tensor
   library under both. Linked statically into one binary, only one copy of each
   ggml symbol survives, so one engine silently ran on the other's ggml version
   (whisper's 0.9 had won, leaving llama.cpp on a ggml older than its own and
   the M5 GPU's neural accelerators disabled for both). Here whisper.cpp builds
   with `WHISPER_USE_SYSTEM_GGML` against the ggml llama-cpp-sys-2 installs and
   exports (`DEP_LLAMA_GGML_CMAKE_DIR`), and links no ggml of its own.
2. **whisper.cpp v1.9.4** (was 1.8.3), a release built for a ggml close to
   llama-cpp-sys-2 0.1.157's (0.23 vs 0.24). Its API only adds to 1.8.3's, so
   whisper-rs 0.16 is unchanged.

`whisper.cpp/` holds only what the build needs from the v1.9.4 tag: `src/`,
`cmake/`, `include/`, `CMakeLists.txt`, `LICENSE` (MIT), and
`bindings/javascript/package-tmpl.json` (read by its CMake configure).
`src/bindings.rs` is the crate's fallback, used only if bindgen fails.

## Updating llama-cpp-2

The ggml comes from llama-cpp-sys-2, so bumping llama-cpp-2 changes the ggml
whisper.cpp runs on. Bump `llama-cpp-sys-2` in `Cargo.toml` here to match, and
move `whisper.cpp/` to the whisper.cpp release whose bundled ggml
(`ggml/CMakeLists.txt`, `GGML_VERSION_MINOR`) is nearest. Then check
transcription still produces text: a mismatched ggml can fail silently — e.g.
llama-cpp-sys-2 0.1.150's ggml had a Metal im2col bug that made Whisper's
encoder output NaN (empty transcripts).
