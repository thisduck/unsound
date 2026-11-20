# transcribe-cpp-sys (patched)

transcribe-cpp-sys 0.4.2 (the crates.io tarball of
[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), MIT, the
engine Handy uses), wired in by `[patch.crates-io]` in `src-tauri/Cargo.toml`.
The app runs Parakeet TDT through it (`src/parakeet.rs`, via the safe
`transcribe-cpp` crate). It builds transcribe.cpp against the ggml
llama-cpp-sys-2 installs, as the vendored whisper-rs-sys does, so the program
keeps one ggml:

1. `CMakeLists.txt`: `TRANSCRIBE_USE_SYSTEM_GGML` does `find_package(ggml)`
   instead of `add_subdirectory(ggml)`. The vendored `ggml/` tree (25 MB) is
   dropped from this copy.
2. `bindings/rust/sys/build.rs`: sets that option with
   `ggml_DIR=$DEP_LLAMA_GGML_CMAKE_DIR/ggml`, `GGML_METAL=ON` (for the Metal
   capability probe), `-DTRANSCRIBE_NO_GGML_REG_FILTER`, and skips the `ggml*`
   archives in the link manifest. `src/lib.rs` names `llama_cpp_sys_2` so its
   archives are linked.
3. `src/transcribe.cpp`: the backend registration filter is compiled out.
   It relies on a transcribe.cpp patch to ggml (`ggml_backend_set_reg_filter`)
   that upstream ggml lacks, and in a shared ggml it would filter llama.cpp's
   backends too. The only loss is the allowed-backend mask (env/API).

transcribe.cpp 0.4.2 vendors ggml 0.26.0 plus four patches; llama-cpp-sys-2
0.1.157 ships ggml 0.24.0. Every ggml symbol libtranscribe references exists
in the 0.24 archives, and on this machine (M5 Max, Metal) Parakeet TDT v2 and
v3 gave byte-identical transcripts on 379 clips built either way. Of the four
patches, three are absent here: the reg filter (above), a threadpool barrier
yield under CPU oversubscription (MSVC/x86 and 2-vCPU runners), and a NULL
check when Metal cannot map a buffer. llama-cpp-sys-2 0.1.161 ships ggml
0.26.0, whose sources differ from transcribe.cpp's only by those patches.

## Updating

Bumping llama-cpp-2 changes the ggml under all three engines — whisper.cpp,
llama.cpp and transcribe.cpp — and a mismatch can fail silently (empty or
garbled transcripts, not errors). Re-verify all three, as described in
`src-tauri/spikes/parakeet-eval/README.md`:

- `examples/asr_eval.rs` with `app-parakeet` (v2 and v3) over the eval sets:
  transcripts non-empty and identical to the standalone `parakeet-eval`
  build, which uses transcribe.cpp's own ggml;
- the same with `whisper`, against a run from the build before the bump;
- `examples/three_engines.rs` with `PARAKEET=app`: Whisper, Parakeet and an
  LLM in one process, checked with `check_three.py`;
- `src/engine_tests.rs` (ignored test): the app's dictation and meeting paths.

When transcribe.cpp moves, re-apply the three patches (marked `unsound:`) and
check that `nm -u libtranscribe.a` needs no ggml symbol the shared ggml lacks.
